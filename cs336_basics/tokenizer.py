from __future__ import annotations

from collections.abc import Iterable


class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]|None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.byte_to_id  = {token_bytes: token_id for token_id, token_bytes in vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
    def encode(self, text: str) -> list[int]:
        # 先把字符串变成 utf-8 bytes
        text_bytes = text.encode("utf-8")

        # 对 bytes 应用 BPE merges
        #
        # 返回的是 list[bytes]
        #
        # 例如：
        # b"abc" 可能变成：
        # [b"abc"]
        #
        # 或者：
        # [b"ab", b"c"]
        bpe_tokens = self._apply_bpe_merges(text_bytes)

        # 把每个 bytes token 查成 token id
        ids = [
            self.byte_to_id[token_bytes]
            for token_bytes in bpe_tokens
        ]

        return ids
    def decode(self, ids: Iterable[int]) -> str:
        """Decodes a list of token ids into a string."""
        token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode('utf-8', errors='replace')
        # 这个函数只处理一段 bytes
    #
    # 输入：
    # b"abc"
    #
    # 初始拆成：
    # [b"a", b"b", b"c"]
    #
    # 然后根据 merges 规则反复合并
    def _apply_bpe_merges(self, text_bytes: bytes) -> list[bytes]:
        # 先把整段 bytes 拆成一个个单 byte token
        tokens = [
            bytes([byte_value])
            for byte_value in text_bytes
        ]

        # 如果长度是 0 或 1，就没有相邻 pair 可以合并
        if len(tokens) <= 1:
            return tokens

        while True:
            best_pair = None
            best_rank = None

            # 找当前 tokens 里面优先级最高的 pair
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])

                # 如果这个 pair 不在 merge_ranks 里，说明它不能合并
                if pair not in self.merge_ranks:
                    continue

                rank = self.merge_ranks[pair]

                # rank 越小，优先级越高
                if best_rank is None or rank < best_rank:
                    best_pair = pair
                    best_rank = rank

            # 如果一个能合并的 pair 都没找到，BPE 结束
            if best_pair is None:
                break

            # 把所有 best_pair 出现的位置都合并
            new_tokens = []
            i = 0

            while i < len(tokens):
                # 如果当前位置和下一个位置组成 best_pair，就合并
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return tokens