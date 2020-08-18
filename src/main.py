#!/usr/bin/env python

import chainer
from teras import training
from teras.app import App, arg
from teras.utils import git, logging
from tqdm import tqdm

import dataset
import eval as eval_module
import models
import parsers
import utils


chainer.Variable.__int__ = lambda self: int(self.data)
chainer.Variable.__float__ = lambda self: float(self.data)
logging.captureWarnings(True)


def train(
        train_file,
        test_file=None,
        format='tree',
        embed_file=None,
        n_epoch=20,
        batch_size=20,
        lr=0.001,
        limit=-1,
        l2_lambda=0.0,
        grad_clip=5.0,
        encoder_input=('char', 'postag'),
        model_config=None,
        device=-1,
        save_dir=None,
        seed=None,
        cache_dir='',
        disable_cache=False,
        refresh_cache=False):
    if seed is not None:
        utils.set_random_seed(seed, device)
    logger = logging.getLogger()
    assert isinstance(logger, logging.AppLogger)
    if model_config is None:
        model_config = {}

    loader = dataset.DataLoader.build(
        word_embed_size=model_config.get('word_embed_size', 100),
        postag_embed_size=model_config.get('postag_embed_size', 50),
        char_embed_size=model_config.get('char_embed_size', 10),
        word_embed_file=embed_file,
        filter_coord=(format == 'tree'),
        enable_cache=not(disable_cache),
        refresh_cache=refresh_cache,
        format=format,
        cache_options=dict(dir=cache_dir, mkdir=True, logger=logger),
        extra_ids=(git.hash(),))

    cont_embed_file_ext = _get_cont_embed_file_ext(encoder_input)
    use_cont_embed = cont_embed_file_ext is not None

    train_dataset = loader.load_with_external_resources(
        train_file, train=True, bucketing=False,
        size=None if limit < 0 else limit, refresh_cache=refresh_cache,
        use_external_postags=True,
        use_contextualized_embed=use_cont_embed,
        contextualized_embed_file_ext=cont_embed_file_ext,
        logger=logger)
    logger.info('{} samples loaded for training'.format(len(train_dataset)))
    test_dataset = None
    if test_file is not None:
        test_dataset = loader.load_with_external_resources(
            test_file, train=False, bucketing=False,
            size=None if limit < 0 else limit // 10,
            refresh_cache=refresh_cache,
            use_external_postags=True,
            use_contextualized_embed=use_cont_embed,
            contextualized_embed_file_ext=cont_embed_file_ext,
            logger=logger)
        logger.info('{} samples loaded for validation'
                    .format(len(test_dataset)))

    builder = models.CoordSolverBuilder(
        loader, inputs=encoder_input, **model_config)
    logger.info("{}".format(builder))
    model = builder.build()
    logger.trace("Model: {}".format(model))
    if device >= 0:
        chainer.cuda.get_device_from_id(device).use()
        model.to_gpu(device)

    optimizer = chainer.optimizers.Adam(
        alpha=lr, beta1=0.9, beta2=0.999, eps=1e-08)
    optimizer.setup(model)
    if l2_lambda > 0.0:
        optimizer.add_hook(chainer.optimizer.WeightDecay(l2_lambda))
    if grad_clip > 0.0:
        optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))

    def _report(y, t):
        values = {}
        model.compute_accuracy(y, t)
        for k, v in model.result.items():
            if 'loss' in k:
                values[k] = float(chainer.cuda.to_cpu(v.data))
            elif 'accuracy' in k:
                values[k] = v
        training.report(values)

    trainer = training.Trainer(optimizer, model, loss_func=model.compute_loss)
    trainer.configure(utils.training_config)
    trainer.add_listener(
        training.listeners.ProgressBar(lambda n: tqdm(total=n)), priority=200)
    trainer.add_hook(
        training.BATCH_END, lambda data: _report(data['ys'], data['ts']))
    if test_dataset:
        parser = parsers.build_parser(loader, model)
        evaluator = eval_module.Evaluator(
            parser, logger=logger, report_details=False)
        trainer.add_listener(evaluator)

    if save_dir is not None:
        accessid = logger.accessid
        date = logger.accesstime.strftime('%Y%m%d')
        metric = 'whole' if isinstance(model, models.Teranishi17) else 'inner'
        trainer.add_listener(utils.Saver(
            model, basename="{}-{}".format(date, accessid),
            context=dict(App.context, builder=builder),
            directory=save_dir, logger=logger, save_best=True,
            evaluate=(lambda _: evaluator.get_overall_score(metric))))

    trainer.fit(train_dataset, test_dataset, n_epoch, batch_size)


def test(model_file, test_file, filter_type=True, limit=-1, device=-1):
    context = utils.Saver.load_context(model_file)
    logger = logging.getLogger()
    logger.trace('# context: {}'.format(context))
    if context.seed is not None:
        utils.set_random_seed(context.seed, device)

    loader = context.builder.loader
    loader.filter_coord = filter_type
    encoder_input = context.encoder_input

    cont_embed_file_ext = _get_cont_embed_file_ext(encoder_input)
    use_cont_embed = cont_embed_file_ext is not None

    test_dataset = loader.load_with_external_resources(
        test_file, train=False, bucketing=False,
        size=None if limit < 0 else limit,
        use_external_postags=True,
        use_contextualized_embed=use_cont_embed,
        contextualized_embed_file_ext=cont_embed_file_ext,
        logger=logger)
    logger.info('{} samples loaded for test'.format(len(test_dataset)))

    model = context.builder.build()
    chainer.serializers.load_npz(model_file, model)
    if device >= 0:
        chainer.cuda.get_device_from_id(device).use()
        model.to_gpu(device)

    parser = parsers.build_parser(loader, model)
    evaluator = eval_module.Evaluator(
        parser, logger=logger, report_details=True)
    reporter = training.listeners.Reporter(logger)

    logger.info('Start decoding')
    utils.chainer_train_off()
    evaluator.on_epoch_validate_begin({'epoch': -1})
    pbar = tqdm(total=len(test_dataset))
    for batch in test_dataset.batch(
            context.batch_size, colwise=True, shuffle=False):
        xs, ts = batch[:-1], batch[-1]
        ys = model.forward(*xs)
        loss = model.compute_loss(ys, ts)
        with reporter:
            values = dict(loss=float(chainer.cuda.to_cpu(loss.data)))
            model.compute_accuracy(ys, ts)
            for k, v in model.result.items():
                if 'loss' in k:
                    values[k] = float(chainer.cuda.to_cpu(v.data))
                elif 'accuracy' in k:
                    values[k] = v
            reporter.report(values)
        evaluator.on_batch_end({'train': False, 'xs': xs, 'ts': ts})
        pbar.update(len(ts))
    pbar.close()
    reporter._output_log("testing", reporter.get_summary(),
                         {'epoch': -1, 'size': len(test_dataset)})
    evaluator.on_epoch_validate_end({'epoch': -1})


def _get_cont_embed_file_ext(encoder_input):
    ext = None
    if sum(('elmo' in encoder_input,
            'bert-base' in encoder_input,
            'bert-large' in encoder_input)) > 1:
        raise ValueError('at most 1 contextualized emebeddings can be chosen')
    elif 'elmo' in encoder_input:
        ext = '.elmo.hdf5'
    elif 'bert-base' in encoder_input:
        ext = '.bert-base.hdf5'
    elif 'bert-large' in encoder_input:
        ext = '.bert-large.hdf5'
    return ext


def parse(model_file, target_file, contextualized_embed_file=None,
          n_best=1, device=-1):
    context = utils.Saver.load_context(model_file)
    logger = logging.getLogger()
    logger.trace('# context: {}'.format(context))
    if context.seed is not None:
        utils.set_random_seed(context.seed, device)

    loader = context.builder.loader
    encoder_input = context.encoder_input
    use_cont_embed = _get_cont_embed_file_ext(encoder_input) is not None
    if use_cont_embed and contextualized_embed_file is None:
        raise ValueError(
            "contextualized_embed_file must be specified when using "
            "a model trained with contextualized embeddings")
    elif not use_cont_embed and contextualized_embed_file is not None:
        raise ValueError(
            "contextualized_embed_file must not be specified when using "
            "a model trained without contextualized embeddings")

    if target_file.endswith('.txt'):
        loader.init_reader(format='default')
    loader.set_contextualized_embed_file(contextualized_embed_file)
    target_dataset = loader.load_with_external_resources(
        target_file, mode='parse', use_external_postags=True, logger=logger)
    logger.info('{} samples loaded for parsing'.format(len(target_dataset)))

    model = context.builder.build()
    chainer.serializers.load_npz(model_file, model)
    if device >= 0:
        chainer.cuda.get_device_from_id(device).use()
        model.to_gpu(device)
    parser = parsers.build_parser(loader, model)

    logger.info('Start parsing')
    utils.chainer_train_off()
    pbar = tqdm(total=len(target_dataset))
    for batch in target_dataset.batch(
            context.batch_size, colwise=True, shuffle=False):
        xs, (words, indices, sentence_id) = batch[:-3], batch[-3:]
        parsed = parser.parse(*xs, n_best)
        for results, words_i, indices_i, sentence_id_i \
                in zip(parsed, words, indices, sentence_id):
            raw_sentence = ' '.join(words_i)
            for best_k, (coords, score) in enumerate(results):
                output = [
                    "#!RAW: {}".format(raw_sentence),
                    "SENTENCE: {}".format(sentence_id_i),
                    "CANDIDATE: #{}".format(best_k),
                    "SCORE: {}".format(score),
                ]
                if indices_i is not None:
                    coords = dataset.postprocess(coords, indices_i)
                for cc, coord in sorted(coords.items()):
                    output.append("CC: {} {}".format(cc, words_i[cc]))
                    if coord is not None:
                        b, e = coord.conjuncts[0][0], coord.conjuncts[-1][1]
                        output.append("COORD: {} {} {}".format(
                            b, e, ' '.join(words_i[b:e + 1])))
                        for (b, e) in coord.conjuncts:
                            output.append("CONJ: {} {} {}".format(
                                b, e, ' '.join(words_i[b:e + 1])))
                    else:
                        output.append("COORD: None")
                print('\n'.join(output) + '\n')
        pbar.update(len(sentence_id))
    pbar.close()


def check_grammar(test_file, limit=-1, grammar_type=1):
    logger = logging.getLogger()
    loader = dataset.DataLoader(filter_coord=True)
    test_dataset = loader.load(test_file, train=True, bucketing=False,
                               size=None if limit < 0 else limit)
    word_vocab = loader.get_processor('word').vocab
    from models.gold import GoldModel
    model = GoldModel()
    if grammar_type == 1:
        cfg = parsers.Grammar.CFG_COORD_1 + parsers.Grammar.CFG
    elif grammar_type == 2:
        cfg = parsers.Grammar.CFG_COORD_2 + parsers.Grammar.CFG
    else:
        raise ValueError("Invalid grammar type: {}".format(grammar_type))
    grammar = parsers.Grammar(word_vocab, cfg)
    parser = parsers.CkyParser(model, grammar)
    evaluator = eval_module.Evaluator(
        parser, logger=logger, report_details=False)
    n_corrects = 0
    pbar = tqdm(total=len(test_dataset))
    for batch in test_dataset.batch(size=20, colwise=True, shuffle=False):
        xs, ts = batch[:-1], batch[-1]
        true_coords_batch = ts
        model.set_gold(true_coords_batch)
        pred_coords_batch = evaluator._parser.parse(*xs, n_best=1)
        for i, (pred_coord_entries, true_coords) in \
                enumerate(zip(pred_coords_batch, true_coords_batch)):
            pred_coords, _score = pred_coord_entries[0]
            true_coords = {ckey: coord for ckey, coord
                           in true_coords.items() if coord is not None}
            for k, v in tuple(pred_coords.items()):
                if v is None:
                    del pred_coords[k]
            if pred_coords == true_coords:
                n_corrects += 1
            else:
                sentence = ' '.join(
                    [word_vocab.lookup(word_id) for word_id in xs[0][i]])
                print("SENTENCE: {}\nPRED: {}\nTRUE: {}\n-"
                      .format(sentence, pred_coords, true_coords))
            evaluator.add(pred_coords, true_coords)
        pbar.update(len(ts))
    pbar.close()
    evaluator.report()
    logger.info("Number of correct tree: {}/{}"
                .format(n_corrects, len(test_dataset)))


if __name__ == "__main__":
    App.configure(logdir=App.basedir + '/../logs')
    logging.AppLogger.configure(mkdir=True)
    App.add_command('train', train, {
        'batch_size':
        arg('--batchsize', type=int, default=20, metavar='NUM',
            help='Number of examples in each mini-batch'),
        'cache_dir':
        arg('--cachedir', type=str, default=(App.basedir + '/../cache'),
            metavar='DIR', help='Cache directory'),
        'disable_cache':
        arg('--nocache', action='store_true', help='Disable cache'),
        'test_file':
        arg('--devfile', type=str, default=None, metavar='FILE',
            help='Development data file'),
        'device':
        arg('--device', type=int, default=-1, metavar='ID',
            help='Device ID (negative value indicates CPU)'),
        'embed_file':
        arg('--embedfile', type=str, default=None, metavar='FILE',
            help='Pretrained word embedding file'),
        'n_epoch':
        arg('--epoch', type=int, default=20, metavar='NUM',
            help='Number of sweeps over the dataset to train'),
        'format':
        arg('--format', type=str, choices=('tree', 'genia'), default='tree',
            help='Training/Development data format'),
        'grad_clip':
        arg('--gradclip', type=float, default=5.0, metavar='VALUE',
            help='L2 norm threshold of gradient norm'),
        'encoder_input':
        arg('--inputs', type=str,
            choices=('char', 'postag', 'elmo', 'bert-base', 'bert-large'),
            nargs='*', default=('char', 'postag'),
            help='Additional inputs for the encoder'),
        'l2_lambda':
        arg('--l2', type=float, default=0.0, metavar='VALUE',
            help='Strength of L2 regularization'),
        'limit':
        arg('--limit', type=int, default=-1, metavar='NUM',
            help='Limit of the number of training samples'),
        'lr':
        arg('--lr', type=float, default=0.001, metavar='VALUE',
            help='Learning Rate'),
        'model_config':
        arg('--model', action='store_dict', metavar='KEY=VALUE',
            help='Model configuration'),
        'refresh_cache':
        arg('--refresh', '-r', action='store_true', help='Refresh cache'),
        'save_dir':
        arg('--savedir', type=str, default=None, metavar='DIR',
            help='Directory to save the model'),
        'seed':
        arg('--seed', type=int, default=None, metavar='VALUE',
            help='Random seed'),
        'train_file':
        arg('--trainfile', type=str, required=True, metavar='FILE',
            help='Training data file'),
    })
    App.add_command('test', test, {
        'filter_type':
        arg('--filter', type=str,
            choices=('any', 'simple', 'not_simple', 'consecutive', 'multiple'),
            default='any', help='Filter type for sentence'),
        'device':
        arg('--device', type=int, default=-1, metavar='ID',
            help='Device ID (negative value indicates CPU)'),
        'limit':
        arg('--limit', type=int, default=-1, metavar='NUM',
            help='Limit of the number of training samples'),
        'model_file':
        arg('--modelfile', type=str, required=True, metavar='FILE',
            help='Trained model file'),
        'test_file':
        arg('--testfile', type=str, required=True, metavar='FILE',
            help='Test data file'),
    })
    App.add_command('parse', parse, {
        'contextualized_embed_file':
        arg('--cembfile', type=str, metavar='FILE',
            help='Contextualized embeddings file'),
        'device':
        arg('--device', type=int, default=-1, metavar='ID',
            help='Device ID (negative value indicates CPU)'),
        'model_file':
        arg('--modelfile', type=str, required=True, metavar='FILE',
            help='Trained model file'),
        'n_best':
        arg('--nbest', type=int, default=1, metavar='NUM',
            help='Number of candidates to output'),
        'target_file':
        arg('--input', type=str, required=True, metavar='FILE',
            help='Input text file to parse'),
    })
    App.add_command('check', check_grammar, {
        'grammar_type':
        arg('--grammar', type=int, choices=(1, 2), default=1,
            help='grammar type'),
        'limit':
        arg('--limit', type=int, default=-1, metavar='NUM',
            help='Limit of the number of training samples'),
        'test_file':
        arg('--testfile', type=str, required=True, metavar='FILE',
            help='Test data file'),
    })
    App.run()
