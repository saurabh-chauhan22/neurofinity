"""
mindmap_evaluation.py — Evaluation Suite for Neuroinfy Mind Map Generation
===========================================================================

This module provides a complete evaluation pipeline for fine-tuned Flan-T5 models
that generate hierarchical mind maps from meeting transcripts. It integrates with
HuggingFace's Seq2SeqTrainer via `compute_metrics` and also supports standalone
evaluation runs.

Three categories of metrics are computed:

1. TEXT SIMILARITY — ROUGE-1/2/L and BLEU to measure surface-level overlap with
   gold-standard mind maps. These are necessary but NOT sufficient for mind maps.

2. STRUCTURAL ACCURACY — Custom metrics that parse the Markdown tree structure and
   compare predicted vs. gold hierarchies. This is the core differentiator: a mind
   map that reads differently but has the correct *structure* is still a good map.

3. VALIDITY CHECKS — Binary checks for well-formedness: Does the output parse as a
   valid tree? Are indentation levels consistent? Is there exactly one root node?

Usage with Seq2SeqTrainer:
    evaluator = MindMapEvaluator(tokenizer)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        compute_metrics=evaluator.compute_metrics,
        ...
    )

Standalone usage:
    evaluator = MindMapEvaluator(tokenizer)
    results = evaluator.evaluate_batch(predictions, references)
    evaluator.print_report(results)

Author: Saurabh (Neuroinfy Project)
"""

import re
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter

import evaluate
from loguru import logger
from transformers import PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# Section 1 — Data Structures for Parsed Mind Map Trees
# ---------------------------------------------------------------------------

@dataclass
class MindMapNode:
    """Represents a single node in a parsed mind map tree.
    
    Each node stores its text content, depth level in the hierarchy,
    and references to its children. This mirrors the MindBench schema
    where each node has a level (0 = root, 1 = main topic, 2+ = subtopics).
    """
    text: str
    level: int
    children: list = field(default_factory=list)
    
    def count_descendants(self) -> int:
        """Recursively count all nodes beneath this one."""
        return sum(1 + child.count_descendants() for child in self.children)
    
    def max_depth(self) -> int:
        """Find the deepest level reachable from this node."""
        if not self.children:
            return self.level
        return max(child.max_depth() for child in self.children)
    
    def get_all_texts(self) -> list[str]:
        """Flatten the tree into a list of all node texts (BFS order)."""
        texts = [self.text]
        for child in self.children:
            texts.extend(child.get_all_texts())
        return texts


# ---------------------------------------------------------------------------
# Section 2 — Markdown Parser for Mind Map Output
# ---------------------------------------------------------------------------

class MindMapParser:
    """Parses hierarchical Markdown into a tree of MindMapNode objects.
    
    Supports two common formats that Flan-T5 might produce:
    
    Format A (Markdown headers):
        # Root Topic
        ## Main Branch 1
        ### Sub-branch 1.1
        ## Main Branch 2
    
    Format B (Indented bullets):
        - Root Topic
          - Main Branch 1
            - Sub-branch 1.1
          - Main Branch 2
    
    The parser auto-detects which format is being used and normalizes both
    into the same MindMapNode tree structure for consistent evaluation.
    """
    
    # Regex patterns for the two supported formats
    HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$')
    BULLET_PATTERN = re.compile(r'^(\s*)-\s+(.+)$')
    
    @staticmethod
    def parse(text: str) -> Optional[MindMapNode]:
        """Parse a mind map string into a tree. Returns None if unparseable.
        
        The parser first tries header format (more structured), then falls
        back to bullet format (more common in freeform generation).
        """
        if not text or not text.strip():
            return None
        
        lines = [line for line in text.strip().split('\n') if line.strip()]
        
        if not lines:
            return None
        
        # Auto-detect format by checking the first non-empty line
        if lines[0].strip().startswith('#'):
            return MindMapParser._parse_header_format(lines)
        elif '-' in lines[0]:
            return MindMapParser._parse_bullet_format(lines)
        else:
            # Last resort: treat the whole text as a single root node.
            # This happens when the model outputs unformatted text.
            return MindMapNode(text=text.strip(), level=0)
    
    @staticmethod
    def _parse_header_format(lines: list[str]) -> Optional[MindMapNode]:
        """Parse Markdown header format (# / ## / ### etc.)."""
        nodes = []
        for line in lines:
            match = MindMapParser.HEADER_PATTERN.match(line.strip())
            if match:
                level = len(match.group(1)) - 1  # '#' = level 0, '##' = level 1, etc.
                text = match.group(2).strip()
                nodes.append(MindMapNode(text=text, level=level))
        
        if not nodes:
            return None
        
        return MindMapParser._build_tree(nodes)
    
    @staticmethod
    def _parse_bullet_format(lines: list[str]) -> Optional[MindMapNode]:
        """Parse indented bullet format (- item / ␣␣- subitem)."""
        nodes = []
        for line in lines:
            match = MindMapParser.BULLET_PATTERN.match(line)
            if match:
                # Each 2 spaces of indentation = 1 depth level
                indent = len(match.group(1))
                level = indent // 2
                text = match.group(2).strip()
                nodes.append(MindMapNode(text=text, level=level))
        
        if not nodes:
            return None
        
        return MindMapParser._build_tree(nodes)
    
    @staticmethod
    def _build_tree(flat_nodes: list[MindMapNode]) -> MindMapNode:
        """Convert a flat list of (text, level) nodes into a proper tree.
        
        Uses a stack-based approach: we maintain a stack of "open" parent nodes.
        When we encounter a node at level N, we pop the stack until the top is
        at level N-1 (our parent), then attach ourselves as a child.
        
        This is the same algorithm used in Markdown-to-AST parsers.
        """
        if not flat_nodes:
            return None
        
        root = flat_nodes[0]
        # The stack holds (node, level) pairs; start with the root
        stack = [root]
        
        for node in flat_nodes[1:]:
            # Pop the stack until we find this node's parent
            # (the most recent node with a lower level)
            while len(stack) > 1 and stack[-1].level >= node.level:
                stack.pop()
            
            # The top of the stack is now our parent
            stack[-1].children.append(node)
            stack.append(node)
        
        return root


# ---------------------------------------------------------------------------
# Section 3 — Structural Metrics (the heart of mind map evaluation)
# ---------------------------------------------------------------------------

class StructuralMetrics:
    """Computes structural similarity between predicted and gold mind map trees.
    
    Why do we need this? Standard text metrics like ROUGE compare token overlap,
    but a mind map's VALUE lies in its structure. Consider:
    
    Gold:    "# Meeting → ## Budget → ### Q1 Numbers → ## Timeline"
    Pred A:  "# Meeting → ## Budget → ### Q1 Figures → ## Timeline"  (good!)
    Pred B:  "# Meeting Budget Q1 Numbers Timeline"                   (bad!)
    
    ROUGE might score Pred B higher (more token overlap), but Pred A is clearly
    the better mind map because it preserves the hierarchical structure.
    """
    
    @staticmethod
    def compute_all(pred_tree: Optional[MindMapNode],
                    gold_tree: Optional[MindMapNode]) -> dict[str, float]:
        """Compute all structural metrics between a predicted and gold tree.
        
        Returns a dictionary with all metric names prefixed with 'struct_' for
        easy identification in training logs.
        """
        # If either tree failed to parse, return zeros across the board.
        # This acts as a penalty for unparseable outputs.
        if pred_tree is None or gold_tree is None:
            return {
                'struct_tree_edit_sim': 0.0,
                'struct_depth_score': 0.0,
                'struct_node_count_ratio': 0.0,
                'struct_branching_sim': 0.0,
                'struct_level_distribution_sim': 0.0,
                'struct_topic_overlap_f1': 0.0,
                'struct_composite': 0.0,
            }
        
        depth_score = StructuralMetrics._depth_similarity(pred_tree, gold_tree)
        node_ratio = StructuralMetrics._node_count_ratio(pred_tree, gold_tree)
        branch_sim = StructuralMetrics._branching_factor_similarity(pred_tree, gold_tree)
        level_sim = StructuralMetrics._level_distribution_similarity(pred_tree, gold_tree)
        topic_f1 = StructuralMetrics._topic_overlap_f1(pred_tree, gold_tree)
        tree_edit = StructuralMetrics._tree_edit_similarity(pred_tree, gold_tree)
        
        # Composite score: weighted average emphasizing hierarchy and topic coverage.
        # These weights reflect what matters most for neurodivergent accessibility:
        # correct depth/branching (so the map is navigable) and topic coverage
        # (so nothing important is missed).
        composite = (
            0.25 * tree_edit +
            0.15 * depth_score +
            0.10 * node_ratio +
            0.15 * branch_sim +
            0.10 * level_sim +
            0.25 * topic_f1
        )
        
        return {
            'struct_tree_edit_sim': round(tree_edit, 4),
            'struct_depth_score': round(depth_score, 4),
            'struct_node_count_ratio': round(node_ratio, 4),
            'struct_branching_sim': round(branch_sim, 4),
            'struct_level_distribution_sim': round(level_sim, 4),
            'struct_topic_overlap_f1': round(topic_f1, 4),
            'struct_composite': round(composite, 4),
        }
    
    @staticmethod
    def _depth_similarity(pred: MindMapNode, gold: MindMapNode) -> float:
        """Compare the maximum depth of both trees.
        
        A mind map that's too shallow loses detail; one that's too deep becomes
        overwhelming. We want the predicted depth to match the gold standard.
        Score of 1.0 means identical depth; decays toward 0 as they diverge.
        """
        pred_depth = pred.max_depth()
        gold_depth = gold.max_depth()
        if gold_depth == 0 and pred_depth == 0:
            return 1.0
        max_depth = max(pred_depth, gold_depth, 1)
        return 1.0 - abs(pred_depth - gold_depth) / max_depth
    
    @staticmethod
    def _node_count_ratio(pred: MindMapNode, gold: MindMapNode) -> float:
        """Ratio of node counts, capped at 1.0 (penalizes both too few and too many).
        
        We use min/max rather than pred/gold because having MORE nodes than the
        gold standard is also a problem (cluttered, overwhelming mind map).
        """
        pred_count = 1 + pred.count_descendants()
        gold_count = 1 + gold.count_descendants()
        return min(pred_count, gold_count) / max(pred_count, gold_count)
    
    @staticmethod
    def _branching_factor_similarity(pred: MindMapNode, gold: MindMapNode) -> float:
        """Compare average branching factors (children per non-leaf node).
        
        The branching factor tells us about the "shape" of the mind map.
        A bushy tree (high branching) vs. a deep narrow tree (low branching)
        convey information very differently. We want these shapes to match.
        """
        def avg_branching(node: MindMapNode) -> float:
            factors = []
            queue = [node]
            while queue:
                current = queue.pop(0)
                if current.children:
                    factors.append(len(current.children))
                    queue.extend(current.children)
            return np.mean(factors) if factors else 0.0
        
        pred_bf = avg_branching(pred)
        gold_bf = avg_branching(gold)
        max_bf = max(pred_bf, gold_bf, 1.0)
        return 1.0 - abs(pred_bf - gold_bf) / max_bf
    
    @staticmethod
    def _level_distribution_similarity(pred: MindMapNode, gold: MindMapNode) -> float:
        """Compare how nodes are distributed across hierarchy levels.
        
        Uses cosine similarity between level-count vectors. For example, if the
        gold map has [1, 3, 5, 2] nodes at levels [0, 1, 2, 3] and the prediction
        has [1, 4, 3, 1], the cosine similarity tells us how well the distribution
        of information across depth levels matches.
        """
        def level_counts(node: MindMapNode) -> Counter:
            counts = Counter()
            queue = [node]
            while queue:
                current = queue.pop(0)
                counts[current.level] += 1
                queue.extend(current.children)
            return counts
        
        pred_counts = level_counts(pred)
        gold_counts = level_counts(gold)
        
        # Union of all levels present in either tree
        all_levels = set(pred_counts.keys()) | set(gold_counts.keys())
        if not all_levels:
            return 1.0
        
        # Build vectors and compute cosine similarity
        pred_vec = np.array([pred_counts.get(l, 0) for l in sorted(all_levels)], dtype=float)
        gold_vec = np.array([gold_counts.get(l, 0) for l in sorted(all_levels)], dtype=float)
        
        dot = np.dot(pred_vec, gold_vec)
        norms = np.linalg.norm(pred_vec) * np.linalg.norm(gold_vec)
        return float(dot / norms) if norms > 0 else 0.0
    
    @staticmethod
    def _topic_overlap_f1(pred: MindMapNode, gold: MindMapNode) -> float:
        """F1 score of topic keywords between predicted and gold trees.
        
        This bridges structural and content evaluation. We extract all node texts,
        tokenize into keywords, and compute precision/recall/F1. This catches cases
        where the structure is right but important topics are missing (low recall)
        or hallucinated topics appear (low precision).
        """
        def extract_keywords(node: MindMapNode) -> set[str]:
            texts = node.get_all_texts()
            keywords = set()
            for text in texts:
                # Simple tokenization: lowercase, split on non-alphanumeric
                tokens = re.findall(r'[a-z0-9]+', text.lower())
                # Filter out very short tokens (articles, prepositions, etc.)
                keywords.update(t for t in tokens if len(t) > 2)
            return keywords
        
        pred_kw = extract_keywords(pred)
        gold_kw = extract_keywords(gold)
        
        if not pred_kw and not gold_kw:
            return 1.0
        if not pred_kw or not gold_kw:
            return 0.0
        
        overlap = pred_kw & gold_kw
        precision = len(overlap) / len(pred_kw)
        recall = len(overlap) / len(gold_kw)
        
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    @staticmethod
    def _tree_edit_similarity(pred: MindMapNode, gold: MindMapNode) -> float:
        """Approximate tree edit distance normalized to a similarity score.
        
        Full Tree Edit Distance (Zhang-Shasha) is O(n^2 * m^2) which is too
        expensive for use inside compute_metrics during training. Instead, we
        use a lightweight approximation based on comparing level-by-level
        sorted node labels, which captures the essential structural alignment
        in O(n log n) time.
        
        The idea: for each level, sort the node texts alphabetically and compute
        the longest common subsequence ratio. Average across all levels.
        """
        def nodes_by_level(node: MindMapNode) -> dict[int, list[str]]:
            levels = {}
            queue = [node]
            while queue:
                current = queue.pop(0)
                levels.setdefault(current.level, []).append(current.text.lower().strip())
                queue.extend(current.children)
            return levels
        
        pred_levels = nodes_by_level(pred)
        gold_levels = nodes_by_level(gold)
        all_levels = set(pred_levels.keys()) | set(gold_levels.keys())
        
        if not all_levels:
            return 1.0
        
        level_scores = []
        for level in sorted(all_levels):
            p_nodes = sorted(pred_levels.get(level, []))
            g_nodes = sorted(gold_levels.get(level, []))
            
            if not p_nodes and not g_nodes:
                level_scores.append(1.0)
            elif not p_nodes or not g_nodes:
                level_scores.append(0.0)
            else:
                # Simple set overlap ratio for this level
                p_set = set(p_nodes)
                g_set = set(g_nodes)
                if p_set | g_set:
                    jaccard = len(p_set & g_set) / len(p_set | g_set)
                    level_scores.append(jaccard)
                else:
                    level_scores.append(0.0)
        
        return float(np.mean(level_scores))


# ---------------------------------------------------------------------------
# Section 4 — Validity Checks (binary quality gates)
# ---------------------------------------------------------------------------

class ValidityChecker:
    """Binary checks for well-formedness of generated mind maps.
    
    These are not scored metrics but pass/fail gates. A mind map that fails
    validity is fundamentally broken regardless of its ROUGE score.
    """
    
    @staticmethod
    def check_all(text: str) -> dict[str, float]:
        """Run all validity checks. Returns 1.0 for pass, 0.0 for fail."""
        return {
            'valid_parseable': float(MindMapParser.parse(text) is not None),
            'valid_has_hierarchy': float(ValidityChecker._has_hierarchy(text)),
            'valid_single_root': float(ValidityChecker._has_single_root(text)),
            'valid_consistent_format': float(ValidityChecker._consistent_format(text)),
            'valid_no_hallucination_markers': float(ValidityChecker._no_hallucination_markers(text)),
        }
    
    @staticmethod
    def _has_hierarchy(text: str) -> bool:
        """Check that the output has at least 2 distinct depth levels.
        A flat list of items is NOT a valid mind map.
        """
        tree = MindMapParser.parse(text)
        if tree is None:
            return False
        levels = set()
        queue = [tree]
        while queue:
            node = queue.pop(0)
            levels.add(node.level)
            queue.extend(node.children)
        return len(levels) >= 2
    
    @staticmethod
    def _has_single_root(text: str) -> bool:
        """A valid mind map has exactly one root node (the central topic)."""
        lines = [l for l in text.strip().split('\n') if l.strip()]
        if not lines:
            return False
        
        # For header format: only one '#' line (single hash)
        root_headers = [l for l in lines if re.match(r'^#\s+', l.strip())]
        if root_headers:
            return len(root_headers) == 1
        
        # For bullet format: only one line at indent level 0
        root_bullets = [l for l in lines if re.match(r'^-\s+', l)]
        if root_bullets:
            return len(root_bullets) == 1
        
        return False
    
    @staticmethod
    def _consistent_format(text: str) -> bool:
        """Check that the output doesn't mix header and bullet formats."""
        lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
        has_headers = any(l.startswith('#') for l in lines)
        has_bullets = any(re.match(r'^\s*-\s+', l) for l in text.split('\n') if l.strip())
        # It's fine to have one or the other, but not both
        return not (has_headers and has_bullets)
    
    @staticmethod
    def _no_hallucination_markers(text: str) -> bool:
        """Check for common signs that the model is hallucinating or confused.
        
        Fine-tuned models sometimes produce repetitive loops, instruction echoing,
        or degenerate outputs. These patterns are red flags.
        """
        red_flags = [
            r'(?:(.{10,}?)\1{3,})',              # Repeated phrases (3+ times)
            r'(?i)generate\s+a\s+mind\s*map',     # Echoing the instruction
            r'(?i)here\s+is\s+(?:a|the)\s+mind',  # Meta-commentary instead of output
            r'(?i)as\s+an\s+ai\s+(?:language\s+)?model',  # AI self-reference
        ]
        for pattern in red_flags:
            if re.search(pattern, text):
                return False
        return True


# ---------------------------------------------------------------------------
# Section 5 — Main Evaluator Class (integrates with Seq2SeqTrainer)
# ---------------------------------------------------------------------------

class MindMapEvaluator:
    """Complete evaluation pipeline for mind map generation.
    
    This is the main class you'll interact with. It wraps all the metrics above
    and provides two interfaces:
    
    1. compute_metrics(eval_preds) — Drop-in for Seq2SeqTrainer
    2. evaluate_batch(predictions, references) — Standalone evaluation
    
    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
        The tokenizer used by your Flan-T5 model. Needed to decode token IDs
        back into text for structural analysis.
    rouge_types : list[str]
        Which ROUGE variants to compute. Default covers unigram, bigram, and
        longest common subsequence.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        rouge_types: list[str] = None,
    ):
        self.tokenizer = tokenizer
        self.rouge_types = rouge_types or ['rouge1', 'rouge2', 'rougeL']
        
        # Load HuggingFace metric modules (these are cached after first load)
        logger.info("Loading evaluation metrics (ROUGE, BLEU)...")
        self.rouge_metric = evaluate.load('rouge')
        self.bleu_metric = evaluate.load('bleu')
        
        self.parser = MindMapParser()
        logger.info("MindMapEvaluator initialized successfully.")
    
    def compute_metrics(self, eval_preds) -> dict[str, float]:
        """Drop-in compute_metrics function for Seq2SeqTrainer.
        
        This is called automatically at the end of each evaluation epoch.
        The eval_preds argument is a named tuple with .predictions and
        .label_ids, both as numpy arrays of token IDs.
        
        IMPORTANT: For this to work, you must set `predict_with_generate=True`
        in your Seq2SeqTrainingArguments. Otherwise, you get raw logits instead
        of generated token sequences, and decoding will fail.
        """
        predictions, labels = eval_preds
        
        # Replace -100 (masked padding tokens) with the pad token ID.
        # HuggingFace uses -100 as an ignore index in the loss function,
        # but the tokenizer can't decode -100.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        
        # Decode token IDs back into strings
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Clean up whitespace (common post-decoding artifact)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        return self.evaluate_batch(decoded_preds, decoded_labels)
    
    def evaluate_batch(
        self,
        predictions: list[str],
        references: list[str],
    ) -> dict[str, float]:
        """Evaluate a batch of predicted mind maps against gold references.
        
        This is the workhorse method. It computes all three categories of
        metrics and returns a flat dictionary suitable for logging to
        wandb, tensorboard, or any other tracker.
        """
        all_metrics = {}
        
        # --- 1. Text Similarity Metrics ---
        all_metrics.update(self._compute_text_metrics(predictions, references))
        
        # --- 2. Structural Metrics (averaged across the batch) ---
        structural_scores = []
        validity_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tree = self.parser.parse(pred)
            ref_tree = self.parser.parse(ref)
            structural_scores.append(StructuralMetrics.compute_all(pred_tree, ref_tree))
            validity_scores.append(ValidityChecker.check_all(pred))
        
        # Average structural metrics across all examples
        for key in structural_scores[0]:
            values = [s[key] for s in structural_scores]
            all_metrics[key] = round(float(np.mean(values)), 4)
        
        # Average validity metrics (these become "pass rates")
        for key in validity_scores[0]:
            values = [s[key] for s in validity_scores]
            all_metrics[key] = round(float(np.mean(values)), 4)
        
        return all_metrics
    
    def _compute_text_metrics(
        self,
        predictions: list[str],
        references: list[str],
    ) -> dict[str, float]:
        """Compute ROUGE and BLEU scores."""
        metrics = {}
        
        # ROUGE scores
        rouge_result = self.rouge_metric.compute(
            predictions=predictions,
            references=references,
            rouge_types=self.rouge_types,
            use_stemmer=True,  # "discussing" matches "discuss"
        )
        for key in self.rouge_types:
            metrics[key] = round(rouge_result[key], 4)
        
        # BLEU score (tokenize into words for each example)
        try:
            tokenized_preds = [pred.split() for pred in predictions]
            tokenized_refs = [[ref.split()] for ref in references]
            bleu_result = self.bleu_metric.compute(
                predictions=tokenized_preds,
                references=tokenized_refs,
            )
            metrics['bleu'] = round(bleu_result['bleu'], 4)
        except ZeroDivisionError:
            # BLEU can fail on empty predictions
            metrics['bleu'] = 0.0
        
        return metrics
    
    def evaluate_single(self, prediction: str, reference: str) -> dict:
        """Evaluate a single prediction with detailed per-metric breakdown.
        
        Useful for debugging specific examples during development.
        """
        pred_tree = self.parser.parse(prediction)
        ref_tree = self.parser.parse(reference)
        
        return {
            'text_metrics': self._compute_text_metrics([prediction], [reference]),
            'structural_metrics': StructuralMetrics.compute_all(pred_tree, ref_tree),
            'validity': ValidityChecker.check_all(prediction),
            'pred_parseable': pred_tree is not None,
            'pred_node_count': (1 + pred_tree.count_descendants()) if pred_tree else 0,
            'pred_max_depth': pred_tree.max_depth() if pred_tree else 0,
            'ref_node_count': (1 + ref_tree.count_descendants()) if ref_tree else 0,
            'ref_max_depth': ref_tree.max_depth() if ref_tree else 0,
        }
    
    def print_report(self, metrics: dict[str, float]) -> str:
        """Format metrics into a readable report string.
        
        Groups metrics by category and highlights the composite score
        which is the single best number to track during training.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("  NEUROINFY — Mind Map Evaluation Report")
        lines.append("=" * 60)
        
        lines.append("\n  TEXT SIMILARITY")
        lines.append("  " + "-" * 40)
        for key in ['rouge1', 'rouge2', 'rougeL', 'bleu']:
            if key in metrics:
                lines.append(f"    {key:<30} {metrics[key]:.4f}")
        
        lines.append("\n  STRUCTURAL ACCURACY")
        lines.append("  " + "-" * 40)
        struct_keys = [k for k in metrics if k.startswith('struct_')]
        for key in struct_keys:
            display = key.replace('struct_', '')
            marker = " ★" if key == 'struct_composite' else ""
            lines.append(f"    {display:<30} {metrics[key]:.4f}{marker}")
        
        lines.append("\n  VALIDITY (pass rates)")
        lines.append("  " + "-" * 40)
        valid_keys = [k for k in metrics if k.startswith('valid_')]
        for key in valid_keys:
            display = key.replace('valid_', '')
            status = "✓" if metrics[key] >= 0.9 else "✗"
            lines.append(f"    {status} {display:<28} {metrics[key]:.1%}")
        
        lines.append("\n" + "=" * 60)
        composite = metrics.get('struct_composite', 0)
        lines.append(f"  COMPOSITE SCORE: {composite:.4f}")
        if composite >= 0.7:
            lines.append("  Status: Strong structural alignment")
        elif composite >= 0.4:
            lines.append("  Status: Moderate — structure captured partially")
        else:
            lines.append("  Status: Weak — model needs more training")
        lines.append("=" * 60)
        
        report = "\n".join(lines)
        logger.info(f"\n{report}")
        return report


# ---------------------------------------------------------------------------
# Section 6 — Integration Example (copy-paste into your training script)
# ---------------------------------------------------------------------------

def get_training_example():
    """Returns a complete Seq2SeqTrainer setup example as a string.
    
    This is not executed, just printed — it serves as documentation
    for how to wire everything together in your training script.
    """
    return '''
# ===== Integration with Seq2SeqTrainer =====

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from mindmap_evaluation import MindMapEvaluator

# 1. Load model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 2. Apply LoRA (keeps training fast and memory-efficient on your H100s)
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,                    # Rank — 16 is a good balance for Flan-T5-base
    lora_alpha=32,           # Scaling factor — typically 2x the rank
    lora_dropout=0.05,       # Light dropout to prevent overfitting on small datasets
    target_modules=["q", "v"],  # Attention query and value projections
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Should be ~0.5-2% of total

# 3. Set up the evaluator
evaluator = MindMapEvaluator(tokenizer)

# 4. Training arguments — tuned for mind map generation
training_args = Seq2SeqTrainingArguments(
    output_dir="./neuroinfy-flan-t5-mindmap",
    
    # === Evaluation settings (critical for our metrics) ===
    eval_strategy="steps",
    eval_steps=200,                # Evaluate every 200 steps
    predict_with_generate=True,    # REQUIRED — enables text generation during eval
    generation_max_length=512,     # Mind maps can be long; allow enough tokens
    
    # === Training hyperparameters ===
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size = 4 * 4 = 16
    learning_rate=3e-4,             # Standard for LoRA fine-tuning
    warmup_ratio=0.06,              # Gentle warmup over 6% of training
    weight_decay=0.01,
    
    # === Logging ===
    logging_steps=50,
    report_to="tensorboard",        # or "wandb" if you prefer
    
    # === Saving ===
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="struct_composite",  # ★ Use OUR metric for model selection!
    greater_is_better=True,
    
    # === Performance ===
    fp16=True,                      # Use mixed precision on H100
    dataloader_num_workers=4,
)

# 5. Data collator handles padding during batching
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,  # Matches what our evaluator expects
)

# 6. Create trainer with our custom evaluator
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,        # Your tokenized dataset
    eval_dataset=eval_dataset,          # Your tokenized eval split
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=evaluator.compute_metrics,  # ★ This is the key line!
)

# 7. Train!
trainer.train()

# 8. After training, run a final evaluation with detailed report
final_results = trainer.evaluate()
evaluator.print_report(final_results)
'''


# ---------------------------------------------------------------------------
# Section 7 — Quick Self-Test (runs when executed directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Running self-test with sample mind maps...\n")
    
    # Sample gold-standard mind map (what we want the model to produce)
    gold = """# Project Status Meeting
## Budget Review
### Q1 Spending
### Q2 Projections
## Timeline Updates
### Phase 1 Complete
### Phase 2 Delays
## Action Items
### Hire contractor
### Update stakeholders"""
    
    # Good prediction — correct structure, slightly different wording
    pred_good = """# Project Status Meeting
## Budget Review
### Q1 Expenditure
### Q2 Forecast
## Timeline Updates
### Phase 1 Done
### Phase 2 Behind Schedule
## Action Items
### Bring on contractor
### Notify stakeholders"""
    
    # Bad prediction — flat structure, missing hierarchy
    pred_bad = """# Project Status Meeting
## Budget Q1 Q2 Timeline Phase 1 Phase 2 Action Items Hire Update"""
    
    # Degenerate prediction — repetitive hallucination
    pred_degenerate = """# Meeting
## Topic
## Topic
## Topic
## Topic
## Topic"""
    
    parser = MindMapParser()
    
    print("\n--- GOOD PREDICTION ---")
    gold_tree = parser.parse(gold)
    good_tree = parser.parse(pred_good)
    good_struct = StructuralMetrics.compute_all(good_tree, gold_tree)
    good_valid = ValidityChecker.check_all(pred_good)
    for k, v in {**good_struct, **good_valid}.items():
        print(f"  {k}: {v}")
    
    print("\n--- BAD PREDICTION (flat structure) ---")
    bad_tree = parser.parse(pred_bad)
    bad_struct = StructuralMetrics.compute_all(bad_tree, gold_tree)
    bad_valid = ValidityChecker.check_all(pred_bad)
    for k, v in {**bad_struct, **bad_valid}.items():
        print(f"  {k}: {v}")
    
    print("\n--- DEGENERATE PREDICTION (repetitive) ---")
    degen_tree = parser.parse(pred_degenerate)
    degen_struct = StructuralMetrics.compute_all(degen_tree, gold_tree)
    degen_valid = ValidityChecker.check_all(pred_degenerate)
    for k, v in {**degen_struct, **degen_valid}.items():
        print(f"  {k}: {v}")
    
    print("\n✓ Self-test complete. Integrate into your Seq2SeqTrainer setup!")