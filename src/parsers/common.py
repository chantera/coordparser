class Parser(object):

    def parse(self, words, postags, chars, cc_indices, sep_indices,
              cont_embeds=None, n_best=1, **kwargs):
        raise NotImplementedError
