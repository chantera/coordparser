#ifndef COORD_CKY_H_
#define COORD_CKY_H_

// #ifdef NDEBUG
// #undef NDEBUG
// #endif

#include <algorithm>
#include <cassert>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using std::int32_t;
using std::size_t;
using std::uint32_t;

#ifndef NDEBUG
#include <iostream>
#define LOG(...) { \
  std::cerr << "[" << __FILE__ << "][" << __FUNCTION__ \
            << "][Line " << __LINE__ << "] " << __VA_ARGS__ << std::endl; \
}
#else
#define LOG(...)
#endif

namespace coord {

struct Coordination {
  struct Conjunct {
    size_t begin;
    size_t end;
  };
  size_t cc;
  std::vector<Conjunct> conjuncts;
  std::vector<size_t> seps;
};

class Dictionary {
 public:
  Dictionary() {}
  ~Dictionary() {}
  Dictionary(const Dictionary &) = delete;
  Dictionary(Dictionary &&) = default;
  Dictionary &operator=(const Dictionary &) = delete;
  Dictionary &operator=(Dictionary &&) = default;

  int32_t add(const std::string &word) {
    std::map<std::string, int32_t>::iterator it = ids_.find(word);
    if (it == ids_.end()) {
      int32_t id = ids_.size();
      ids_.insert(std::make_pair(word, id));
      words_.push_back(word);
      return id;
    }
    return it->second;
  }

  int32_t get(const std::string &word) const {
    std::map<std::string, int32_t>::const_iterator it = ids_.find(word);
    return it == ids_.end() ? -1 : it->second;
  }

  std::string lookup(int32_t id) const {
    if (id < 0 || id >= static_cast<int32_t>(words_.size())) {
      std::stringstream ss;
      ss << "Dictionary: invalid word ID: " << id;
      throw std::out_of_range(ss.str());
    }
    return words_[id];
  }

  inline size_t size() const { return ids_.size(); }
  inline const std::vector<std::string> &get_words() const { return words_; }

 private:
  std::map<std::string, int32_t> ids_;
  std::vector<std::string> words_;
};

class Grammar {
 public:
  struct LexicalEntry {
    enum Type {
      NORMAL = 0,
      ANY = 1,
      CC = 2,
      SEP = 3,
    };
    uint32_t parent;
    uint32_t item;
    Type type;

    bool operator==(const LexicalEntry &rhs) const {
      return (parent == rhs.parent && item == rhs.item);
    }
  };
  struct UnaryRule {
    uint32_t parent;
    uint32_t child;

    bool operator==(const UnaryRule &rhs) const {
      return (parent == rhs.parent && child == rhs.child);
    }
  };
  struct BinaryRule {
    uint32_t parent;
    uint32_t left;
    uint32_t right;

    bool operator==(const BinaryRule &rhs) const {
      return (parent == rhs.parent && left == rhs.left && right == rhs.right);
    }
  };

  Grammar() {}
  ~Grammar() {}
  Grammar(const Grammar &) = delete;
  Grammar(Grammar &&) = delete;
  Grammar &operator=(const Grammar &) = delete;
  Grammar &operator=(Grammar &&) = delete;

  void add_lexicon(const std::string &parent, uint32_t item, int32_t type = 0) {
    const LexicalEntry entry = { add_tag(parent), item, static_cast<LexicalEntry::Type>(type) };
    if (std::find(lexicon_.begin(), lexicon_.end(), entry) == lexicon_.end()) {
      lexicon_.emplace_back(std::move(entry));
    }
  }
  void add_unary(const std::string &parent, const std::string &child) {
    const UnaryRule u_rule = { add_tag(parent), add_tag(child) };
    if (std::find(unary_rules_.begin(), unary_rules_.end(), u_rule) == unary_rules_.end()) {
      unary_rules_.emplace_back(std::move(u_rule));
    }
  }
  void add_binary(const std::string &parent, const std::string &left, const std::string &right) {
    const BinaryRule b_rule = { add_tag(parent), add_tag(left), add_tag(right) };
    if (std::find(binary_rules_.begin(), binary_rules_.end(), b_rule) == binary_rules_.end()) {
      binary_rules_.emplace_back(std::move(b_rule));
    }
  }

  const Dictionary &get_tagset() const { return tagset_; }
  const std::vector<LexicalEntry> &get_lexicon() const { return lexicon_; }
  const std::vector<UnaryRule> &get_unary_rules() const { return unary_rules_; }
  const std::vector<BinaryRule> &get_binary_rules() const { return binary_rules_; }

 private:
  uint32_t add_tag(const std::string &tag) { return static_cast<uint32_t>(tagset_.add(tag)); }

  Dictionary tagset_;
  std::vector<LexicalEntry> lexicon_;
  std::vector<UnaryRule> unary_rules_;
  std::vector<BinaryRule> binary_rules_;
};

template <class T>
class CkyTable {
 public:
  CkyTable(size_t num_words, size_t num_tags)
    : num_words_(num_words), num_tags_(num_tags) {
    if (num_words_ == 0 || num_tags_ == 0) throw std::runtime_error("CkyTable: invalid table size");
    shift_ = new uint32_t[num_words_ + 1];
    shift_[0] = 0;
    shift_[1] = 0;
    for (size_t i = 2; i <= num_words_; ++i) {
      shift_[i] = shift_[i - 1] + (num_words_ - i + 2);
    }
    cells_ = new T[num_tags_ * ((num_words_ * (num_words_ + 1)) >> 1)];
  }

  ~CkyTable() {
    delete[] shift_;
    delete[] cells_;
  }

  CkyTable(const CkyTable &) = delete;
  CkyTable(CkyTable &&) = delete;
  CkyTable &operator=(const CkyTable &) = delete;
  CkyTable &operator=(CkyTable &&) = delete;

  inline const T &at(size_t begin, size_t end, size_t tag) const {
    if (begin >= num_words_ || end > num_words_ || end <= begin || tag >= num_tags_) {
      throw std::runtime_error("CkyTable: invalid index");
    }
    return cells_[num_tags_ * (shift_[end - begin] + begin) + tag];
  }

  inline T &at(size_t begin, size_t end, size_t tag) {
    return const_cast<T &>(static_cast<const CkyTable &>(*this).at(begin, end, tag));
  }

  inline size_t num_words() const { return num_words_; }
  inline size_t num_tags() const { return num_tags; }

 private:
  size_t num_words_;
  size_t num_tags_;
  uint32_t* shift_;
  T* cells_;
};

class CkyNode {
 public:
  enum Type {
    LEXICAL = 0,
    UNARY = 1,
    BINARY = 2,
  };
  union BackPointer {
    struct {
      size_t prev_tag;
      size_t prev_agenda;
    };
    struct {
      size_t prev_mid;
      size_t prev_left_tag;
      size_t prev_right_tag;
      size_t prev_left_agenda;
      size_t prev_right_agenda;
    };
  };

  // Lexical
  explicit CkyNode(float score) : type_(LEXICAL), bp_(), score_(score), coord_() {}

  // Unary
  CkyNode(float score, size_t prev_tag, size_t prev_agenda) : type_(UNARY), bp_(), score_(score), coord_() {
    bp_.prev_tag = prev_tag;
    bp_.prev_agenda = prev_agenda;
  }

  // Binary
  CkyNode(float score, size_t prev_mid, size_t prev_left_tag, size_t prev_right_tag,
          float prev_left_agenda, float prev_right_agenda)
    : CkyNode(score, prev_mid, prev_left_tag, prev_right_tag, prev_left_agenda, prev_right_agenda, nullptr) {}

  // Binary with coordination
  CkyNode(float score, size_t prev_mid, size_t prev_left_tag, size_t prev_right_tag,
          float prev_left_agenda, float prev_right_agenda, std::unique_ptr<Coordination> coord)
    : type_(BINARY), bp_(), score_(score), coord_(std::move(coord)) {
    bp_.prev_mid = prev_mid;
    bp_.prev_left_tag = prev_left_tag;
    bp_.prev_right_tag = prev_right_tag;
    bp_.prev_left_agenda = prev_left_agenda;
    bp_.prev_right_agenda = prev_right_agenda;
  }

  ~CkyNode() {}
  CkyNode(const CkyNode &other)
    : type_(other.type_), bp_(other.bp_), score_(other.score_), coord_(other.coord_ ? new Coordination { *other.coord_ } : nullptr) {}
  CkyNode(CkyNode &&other)
    : type_(other.type_), bp_(std::move(other.bp_)), score_(other.score_), coord_(std::move(other.coord_)) {}
  CkyNode &operator=(const CkyNode &other) {
    if (this != &other) {
      if (type_ != other.type_) throw std::runtime_error("CkyNode: cannot copy to a different type of node");
      bp_ = other.bp_;
      score_ = other.score_;
      coord_.reset(other.coord_ ? new Coordination { *other.coord_ } : nullptr);
    }
    return *this;
  }
  CkyNode &operator=(CkyNode &&other) {
    if (this != &other) {
      if (type_ != other.type_) throw std::runtime_error("CkyNode: cannot move to a different type of node");
      bp_ = std::move(other.bp_);
      score_ = other.score_;
      coord_ = std::move(other.coord_);
    }
    return *this;
  }

  Type type() const { return type_; }
  const BackPointer &bp() const { return bp_; }
  float score() const { return score_; }
  const Coordination *const coord() const { return coord_.get(); }

 private:
  Type type_;
  BackPointer bp_;
  float score_;
  std::unique_ptr<Coordination> coord_;
};

template<typename T>
std::string join(const T* array, size_t length, const char* delimiter) {
  std::stringstream ss;
  for (size_t i = 0; i < length; ++i) {
    if (i != 0) {
      ss << delimiter;
    }
    ss << array[i];
  }
  return ss.str();
}

typedef std::string ScoreTableKey;

struct ScoreTableValue {
  const size_t ckey;
  const std::pair<size_t, size_t> left_conj;
  const std::pair<size_t, size_t> right_conj;
  float score;
};

typedef std::map<ScoreTableKey, ScoreTableValue> ScoreTable;

ScoreTableKey gen_key(
    size_t begin, size_t mid, size_t end,
    const Grammar::BinaryRule &b_rule,
    size_t left_agenda, size_t right_agenda) {
  const std::vector<size_t> seed = {
    begin, mid, end, left_agenda, right_agenda, b_rule.parent, b_rule.left, b_rule.right,
  };
  return join(&seed[0], seed.size(), "-");
}

std::pair<size_t, std::pair<size_t, size_t>> find_cc_and_right_conj(
    const CkyTable<std::vector<CkyNode>> &table, size_t begin, size_t end, size_t tag, size_t agenda,
    uint32_t cc_tag, uint32_t conj_tag) {
  const CkyNode &node = table.at(begin, end, tag)[agenda];
  if (node.type() != CkyNode::Type::BINARY) {
    throw std::runtime_error("find_cc_and_right_conj: invalid node type");
  }
  const CkyNode::BackPointer &bp = node.bp();
  size_t mid = bp.prev_mid;
  bool found_cc = false;
  bool found_conj = false;
  uint32_t cc = 0;
  std::pair<size_t, size_t> conj = std::make_pair(0, 0);
  if (bp.prev_left_tag == cc_tag) {
    assert(begin == mid - 1);
    cc = begin;
    found_cc = true;
  }
  if (bp.prev_right_tag == conj_tag) {
    conj = std::make_pair(mid, end - 1);
    found_conj = true;
  }
  if (found_conj) {
    return std::make_pair(cc, conj);
  } else if (found_cc) {
    conj = find_cc_and_right_conj(table, mid, end,
        bp.prev_right_tag, bp.prev_right_agenda, cc_tag, conj_tag).second;
    return std::make_pair(cc, conj);
  } else {
    return find_cc_and_right_conj(table, mid, end,
        bp.prev_right_tag, bp.prev_right_agenda, cc_tag, conj_tag);
  }
}

void traverse(
    const CkyTable<std::vector<CkyNode>> &table, size_t begin, size_t end, size_t tag, size_t agenda,
    std::vector<Coordination> &coords) {
  const CkyNode &node = table.at(begin, end, tag)[agenda];
  const CkyNode::BackPointer &bp = node.bp();
  if (node.type() == CkyNode::Type::BINARY) {
    const Coordination *coord = node.coord();
    if (coord) {
      bool exists = false;
      for (const auto &c : coords) {
        if (coord->cc == c.cc) {
          exists = true;
          break;
        }
      }
      if (!exists) {
        coords.emplace_back(*coord);
      }
    }
    traverse(table, begin, bp.prev_mid, bp.prev_left_tag, bp.prev_left_agenda, coords);
    traverse(table, bp.prev_mid, end, bp.prev_right_tag, bp.prev_right_agenda, coords);
  } else if (node.type() == CkyNode::Type::UNARY) {
    traverse(table, begin, end, bp.prev_tag, bp.prev_agenda, coords);
  }
}

std::vector<std::pair<std::vector<Coordination>, float>> table2coords(
    const CkyTable<std::vector<CkyNode>> &table, size_t begin, size_t end, size_t tag) {
  std::vector<std::pair<std::vector<Coordination>, float>> n_best_coords;
  const auto &children = table.at(begin, end, tag);
  n_best_coords.reserve(children.size());
  for (size_t i = 0; i < children.size(); ++i) {
    std::vector<Coordination> coords;
    traverse(table, begin, end, tag, i, coords);
    n_best_coords.emplace_back(std::make_pair(std::move(coords), children[i].score()));
  }
  return n_best_coords;
}

class CkyParser {
 public:
  explicit CkyParser(const Grammar &grammar)
    : CkyParser(grammar,
                grammar.get_tagset().get("S"),
                grammar.get_tagset().get("COORD"),
                grammar.get_tagset().get("CJT"),
                grammar.get_tagset().get("CC"),
                grammar.get_tagset().get("CC-SEP")) {}

  CkyParser(
    const Grammar &grammar,
    uint32_t tag_complete,
    uint32_t tag_coord,
    uint32_t tag_conj,
    uint32_t tag_cc,
    uint32_t tag_sep)
    : grammar_(&grammar)
    , table_(nullptr)
    , level_(0)
    , score_table_()
    , n_best_(0)
    , tag_complete_(tag_complete)
    , tag_coord_(tag_coord)
    , tag_conj_(tag_conj)
    , tag_cc_(tag_cc)
    , tag_sep_(tag_sep) {}

  ~CkyParser() {}
  CkyParser(const CkyParser &) = delete;
  CkyParser(CkyParser &&) = delete;
  CkyParser &operator=(const CkyParser &) = delete;
  CkyParser &operator=(CkyParser &&) = delete;

  void start_parsing(
      const std::vector<uint32_t> &words, const std::vector<std::pair<float, float>> &ckey_scores, size_t n_best = 1) {
    if (n_best < 1) throw std::runtime_error("CkyParser: `n_best` must be greater than 0");
    table_.reset(new CkyTable<std::vector<CkyNode>>(words.size(), grammar_->get_tagset().size()));
    level_ = 1;
    score_table_.clear();
    n_best_ = n_best;
    LOG("started: level=" << level_);
    LOG("process_lexicon");
    process_lexicon(words, ckey_scores);
    LOG("process_unary");
    process_unary();
    ++level_;
    if (!finished()) {
      LOG("preprocess_binary");
      preprocess_binary();
    }
  }

  std::vector<std::pair<std::vector<Coordination>, float>> get_results() const {
    if (!finished()) throw std::runtime_error("CkyParser: could not retrive results");
    return table2coords(*table_, 0, table_->num_words(), tag_complete_);
  }

  void resume() {
    if (finished()) return;
    LOG("resumed: level=" << level_);
    LOG("process_binary");
    process_binary();
    LOG("process_unary");
    process_unary();
    ++level_;
    if (!finished()) {
      LOG("preprocess_binary");
      preprocess_binary();
    } else {
      LOG("finished");
    }
  }

  inline bool finished() const {
    return level_ > table_->num_words();
  }

  const ScoreTable &get_score_table() const { return score_table_; }

  void update_score_table(const std::vector<std::pair<ScoreTableKey, float>> &kvs) {
    for (const auto &kv : kvs) {
      ScoreTable::iterator it = score_table_.find(kv.first);
      if (it == score_table_.end()) throw std::runtime_error("CkyParser: invalid key");
      it->second.score = kv.second;
    }
  }

 private:
  void process_lexicon(
      const std::vector<uint32_t> &words, const std::vector<std::pair<float, float>> &ckey_scores) {
    if (finished()) return;
    for (size_t begin = 0; begin < words.size(); ++begin) {
      size_t end = begin + 1;
      bool can_be_ckey = false;
      for (const Grammar::LexicalEntry &entry : grammar_->get_lexicon()) {
        if ((entry.type == Grammar::LexicalEntry::CC || entry.type == Grammar::LexicalEntry::SEP)
            && entry.item == words[begin]) {
          can_be_ckey = true;
          break;
        }
      }
      for (const Grammar::LexicalEntry &entry : grammar_->get_lexicon()) {
        if (entry.type == Grammar::LexicalEntry::ANY || entry.item == words[begin]) {
          float lexicon_score = 0.0;
          if (can_be_ckey) {
            const std::pair<float, float> &score_entry = ckey_scores[begin];
            bool is_ckey = (entry.type == Grammar::LexicalEntry::CC || entry.type == Grammar::LexicalEntry::SEP);
            lexicon_score = is_ckey ? score_entry.second : score_entry.first;
          }
          table_->at(begin, end, entry.parent).emplace_back(CkyNode(lexicon_score));
        }
      }
    }
  }

  void process_unary() {
    if (finished()) return;
    size_t n = table_->num_words();
    for (size_t begin = 0; begin < n - level_ + 1; ++begin) {
      size_t end = begin + level_;
      for (const Grammar::UnaryRule &u_rule : grammar_->get_unary_rules()) {
        if (u_rule.parent == tag_complete_ && level_ != n) continue;
        auto &children = table_->at(begin, end, u_rule.child);
        std::vector<CkyNode> nodes;
        nodes.reserve(children.size());
        size_t index = 0;
        std::transform(children.begin(), children.end(), std::back_inserter(nodes), [&u_rule, &index](const CkyNode &child) {
          float unary_score = 0.0;
          return CkyNode(unary_score + child.score(), u_rule.child, index++);
        });
        update_cell(table_->at(begin, end, u_rule.parent), nodes, n_best_, false);
      }
    }
  }

  void preprocess_binary() {
    if (level_ < 2 || finished()) return;
    score_table_.clear();
    size_t n = table_->num_words();
    for (size_t begin = 0; begin < n - level_ + 1; ++begin) {
      size_t end = begin + level_;
      for (size_t mid = begin + 1; mid < end; ++mid) {
        for (const Grammar::BinaryRule &b_rule : grammar_->get_binary_rules()) {
          if (b_rule.parent != tag_coord_) continue;
          assert(b_rule.left == tag_conj_);
          auto &left_children = table_->at(begin, mid, b_rule.left);
          auto &right_children = table_->at(mid, end, b_rule.right);
          if (left_children.size() == 0 || right_children.size() == 0) continue;
          for (size_t i = 0; i < left_children.size(); ++i) {
            const std::pair<size_t, size_t> left_conj = std::make_pair(begin, mid - 1);
            for (size_t j = 0; j < right_children.size(); ++j) {
              std::pair<size_t, size_t> right_conj; 
              size_t ckey;
              const Coordination *const right_coord = right_children[j].coord();
              if (right_coord) {
                // more than 2 consecutive conjuncts
                assert(left_conj.second < right_coord->cc && right_coord->cc < right_coord->conjuncts.back().begin);
                right_conj = std::make_pair(right_coord->conjuncts[0].begin, right_coord->conjuncts[0].end);
                assert(left_conj.second + 2 == right_conj.first);
                ckey = left_conj.second + 1;
              } else {
                // exact 2 conjuncts
                const auto cc_and_right_conj = 
                  find_cc_and_right_conj(*table_, mid, end, b_rule.right, j, tag_cc_, tag_conj_);
                const size_t cc = cc_and_right_conj.first;
                right_conj = cc_and_right_conj.second;
                assert(left_conj.second < cc && cc < right_conj.first);
                ckey = cc;
              }
              float initial_score = -std::numeric_limits<float>::infinity();
              const ScoreTableKey table_key = gen_key(begin, mid, end, b_rule, i, j);
              ScoreTable::iterator it = score_table_.find(table_key);
              if (it != score_table_.end()) throw std::runtime_error("CkyParser: key collides");
              score_table_.insert(
                  std::make_pair(table_key, ScoreTableValue { ckey, left_conj, right_conj, initial_score }));
            }
          }
        }
      }
    }
  }

  void process_binary() {
    if (level_ < 2 || finished()) return;
    size_t n = table_->num_words();
    for (size_t begin = 0; begin < n - level_ + 1; ++begin) {
      size_t end = begin + level_;
      for (size_t mid = begin + 1; mid < end; ++mid) {
        for (const Grammar::BinaryRule &b_rule : grammar_->get_binary_rules()) {
          if (b_rule.parent == tag_complete_ && level_ != n) continue;
          auto &left_children = table_->at(begin, mid, b_rule.left);
          auto &right_children = table_->at(mid, end, b_rule.right);
          size_t n_next_nodes = left_children.size() * right_children.size();
          if (n_next_nodes == 0) continue;
          std::vector<CkyNode> nodes;
          nodes.reserve(n_next_nodes);
          for (size_t i = 0; i < left_children.size(); ++i) {
            for (size_t j = 0; j < right_children.size(); ++j) {
              generate_binary_node(nodes, begin, mid, end, b_rule, i, j, left_children[i], right_children[j]);
            }
          }
          update_cell(table_->at(begin, end, b_rule.parent), nodes, n_best_, false);
        }
      }
    }
  }

  void generate_binary_node(
    std::vector<CkyNode> &nodes,
    size_t begin, size_t mid, size_t end,
    const Grammar::BinaryRule &b_rule,
    size_t left_agenda, size_t right_agenda,
    const CkyNode &left_child, const CkyNode &right_child) const {
      float binary_score = 0.0;
      std::unique_ptr<Coordination> coord;
      if (b_rule.parent == tag_coord_) {
        const auto &value = score_table_.at(gen_key(begin, mid, end, b_rule, left_agenda, right_agenda));
        binary_score = value.score;
        assert(!left_child.coord());
        const Coordination *right_coord = right_child.coord();
        if (right_coord) {
          coord.reset(new Coordination { *right_coord });  // copy
          coord->conjuncts.insert(coord->conjuncts.begin(), Coordination::Conjunct { begin, mid - 1 });
          coord->seps.insert(coord->seps.begin(), mid);
        } else {
          const Coordination::Conjunct left_conj = { begin, mid - 1 };
          const auto cc_and_right_conj = find_cc_and_right_conj(
              *table_, mid, end, b_rule.right, right_agenda, tag_cc_, tag_conj_);
          const size_t cc = cc_and_right_conj.first;
          const Coordination::Conjunct right_conj = { cc_and_right_conj.second.first, cc_and_right_conj.second.second };
          assert(left_conj.end < cc && cc < right_conj.begin);
          coord.reset(new Coordination { cc, { left_conj, right_conj } });
        }
      } else {
        if (b_rule.left == tag_sep_ && b_rule.right == tag_coord_) {
          assert(right_child.coord());
          coord.reset(new Coordination { *right_child.coord() });
        } else {
          // pass
        }
      }
      float score = binary_score + left_child.score() + right_child.score();
      nodes.emplace_back(CkyNode(score, mid, b_rule.left, b_rule.right, left_agenda, right_agenda, std::move(coord)));
  }

  void update_cell(std::vector<CkyNode> &cell, std::vector<CkyNode> &nodes, size_t max_size, bool keep_tie = false) {
    const size_t MAX_ADDITIONAL_SIZE = 32 - max_size;
    if (max_size < 1) throw std::runtime_error("CkyParser: `max_size` must be greater than 0");
    if (nodes.size() == 0) return;
    std::sort(nodes.begin(), nodes.end(), [](const CkyNode &a, const CkyNode &b) {
      return b.score() < a.score();
    });
    size_t additional_size = 0;
    if (nodes.size() > max_size) {
      if (keep_tie) {
        float best_score = nodes[0].score();
        for (size_t i = 2; i <= nodes.size(); ++i) {
          if (nodes[i].score() != best_score || additional_size >= MAX_ADDITIONAL_SIZE) {
            break;
          }
          ++additional_size;
        }
      }
      nodes.erase(nodes.begin() + max_size + additional_size, nodes.end());
    }
    assert(nodes.size() >= 1 && nodes.size() <= (max_size + additional_size));
    // additional_size = 0;
    if (cell.size() == 0) {
      cell.insert(cell.end(), nodes.begin(), nodes.end());
    } else {
      float best_score = nodes[0].score();
      if (cell.back().score() >= best_score) {
        cell.insert(cell.end(), nodes.begin(), nodes.end());
      } else {
        for (auto it = cell.begin(); it != cell.end(); ++it) {
          if (it->score() < best_score) {
            cell.insert(it, nodes.begin(), nodes.end());
            break;
          }
        }
      }
      additional_size = 0;
      if (cell.size() > max_size) {
        if (keep_tie) {
          best_score = cell[0].score();
          for (size_t i = 2; i <= cell.size(); ++i) {
            if (cell[i].score() != best_score || additional_size >= MAX_ADDITIONAL_SIZE) {
              break;
            }
            ++additional_size;
          }
          LOG("additional_size: " << additional_size);
        }
        cell.erase(cell.begin() + max_size + additional_size, cell.end());
      }
      assert(cell.size() >= 1 && cell.size() <= (max_size + additional_size));
      LOG("cell size: " << cell.size());
    }
  }

  const Grammar* grammar_;
  std::unique_ptr<CkyTable<std::vector<CkyNode>>> table_;
  uint32_t level_;
  ScoreTable score_table_;
  size_t n_best_;
  const uint32_t tag_complete_;
  const uint32_t tag_coord_;
  const uint32_t tag_conj_;
  const uint32_t tag_cc_;
  const uint32_t tag_sep_;
};

}  // namespace coord

#undef LOG

#endif  // COORD_CKY_H_
