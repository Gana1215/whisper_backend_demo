import re


class NumberResolver:
    def __init__(self):
        self.vocab = [
            "тэг", "нэг", "хоёр", "гурав", "дөрөв", "тав", "зургаа", "долоо", "найм", "ес",
            "арав", "арван", "хорь", "хорин", "гуч", "гучин", "дөч", "дөчин", "тавь", "тавин",
            "жар", "жаран", "дал", "далан", "ная", "наян", "ер", "ерэн",
            "зуу", "зуун", "мянга", "мянган", "түм", "түмэн", "бум", "буман", "сая", "тэрбум"
        ]

        self.values = {
            "тэг": 0, "нэг": 1, "нэгэн": 1, "хоёр": 2, "хоёрон": 2, "гурав": 3, "гурван": 3,
            "дөрөв": 4, "дөрвөн": 4, "тав": 5, "таван": 5, "зургаа": 6, "зургаан": 6,
            "долоо": 7, "долоон": 7, "найм": 8, "найман": 8, "ес": 9, "есөн": 9,
            "арав": 10, "арван": 10, "хорь": 20, "хорин": 20, "гуч": 30, "гучин": 30,
            "дөч": 40, "дөчин": 40, "тавь": 50, "тавин": 50, "жар": 60, "жаран": 60,
            "дал": 70, "далан": 70, "ная": 80, "наян": 80, "ер": 90, "ерэн": 90,
            "зуу": 100, "зуун": 100, "мянга": 1000, "мянган": 1000, "түм": 10000,
            "түмэн": 10000, "бум": 100000, "буман": 100000, "сая": 1000000, "тэрбум": 1000000000
        }

        self.vowels = set("аэиоуөүяеёю")

        self._hundreds = {"зуу", "зуун"}
        self._tens = {
            "арав", "арван", "хорь", "хорин", "гуч", "гучин", "дөч", "дөчин",
            "тавь", "тавин", "жар", "жаран", "дал", "далан", "ная", "наян", "ер", "ерэн"
        }
        self._scales_big = {"мянга", "мянган", "түм", "түмэн", "бум", "буман", "сая", "тэрбум"}

        # For glued token splitting: use ALL known number-words, longest-first
        self._dict_words = sorted(self.values.keys(), key=len, reverse=True)

        # ⭐ marker words we will split around, longest-first
        self._split_markers = sorted(self._scales_big | self._hundreds, key=len, reverse=True)

        # ⭐ JOIN PATCH (applies to ALL digit tokens algorithmically, not hardcoded per-word):
        # if pattern "<digit> нь" appears, turn it into Mongolian connective form (…ан/…өн/…эн)
        # and DROP "нь" so it never reaches Levenshtein.
        self._particle_n = "нь"
        self._ones_stems = {"нэг", "хоёр", "гурав", "дөрөв", "тав", "зургаа", "долоо", "найм", "ес"}

    # ---------------------------
    # Weighted Levenshtein
    # ---------------------------
    def _weighted_dist(self, s1, s2):
        n, m = len(s1), len(s2)
        dp = [[0.0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i * 1.1
        for j in range(m + 1):
            dp[0][j] = j * 1.1

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                c1, c2 = s1[i - 1], s2[j - 1]
                if c1 == c2:
                    cost = 0.0
                else:
                    cost = 0.5 if (c1 in self.vowels and c2 in self.vowels) else 1.1

                dp[i][j] = min(
                    dp[i - 1][j] + 1.1,
                    dp[i][j - 1] + 1.1,
                    dp[i - 1][j - 1] + cost
                )
        return dp[n][m]

    # ---------------------------
    # Fix single token -> nearest vocab
    # ---------------------------
    def _fix(self, word):
        w_low = (word or "").lower()
        clean = re.sub(r"[^а-яөүё]", "", w_low)
        if not clean or len(clean) < 2:
            return None

        # If already known, keep
        if clean in self.values:
            return clean

        best, min_d = None, float("inf")
        for cand in self.vocab:
            d = self._weighted_dist(clean, cand)
            if d < min_d:
                min_d, best = d, cand
        return best

    # ---------------------------
    # Split token around scale markers anywhere inside
    # ---------------------------
    def _split_by_markers(self, s: str):
        """
        Forces split around known scale markers (зуу/зуун/мянга/...) anywhere inside.
        Examples:
          'зууггурав'   -> ['зуу', 'ггурав']
          'abcзууdef'   -> ['abc', 'зуу', 'def']
          'мянгатаван'  -> ['мянга', 'таван']
        """
        out = []
        i = 0
        n = len(s)

        while i < n:
            hit = None
            for mk in self._split_markers:
                if s.startswith(mk, i):
                    hit = mk
                    break

            if hit:
                out.append(hit)
                i += len(hit)
                continue

            j = i + 1
            while j <= n:
                if any(s.startswith(mk, j) for mk in self._split_markers):
                    break
                j += 1

            chunk = s[i:j]
            if chunk:
                out.append(chunk)
            i = j

        return [x for x in out if x]

    # ---------------------------
    # ⭐ JOIN PATCH: convert "<digit> нь" to connective numeric form, drop "нь"
    # ---------------------------
    def _to_connective_form(self, stem: str):
        """
        stem is expected to be a 1..9 word like "гурав", "тав", "дөрөв" etc.
        Return Mongolian connective form used before scales (…ан/…өн/…эн),
        using a small phonological rule + a couple of special cases.
        """
        stem = stem.lower()

        # special/common literary forms
        if stem == "нэг":
            return "нэгэн"
        if stem == "хоёр":
            return "хоёр"     # "хоёр зуу" is acceptable
        if stem == "гурав":
            return "гурван"

        # very common suffix pattern:
        # - stems ending with "в" or "р" often take "өн/ан"
        # - stems ending with vowel-like often take "н"
        # We'll keep it simple & safe for 1..9.
        # These are already in values mapping, so returning one of those keys is critical.
        if stem == "дөрөв":
            return "дөрвөн"
        if stem == "тав":
            return "таван"
        if stem == "зургаа":
            return "зургаан"
        if stem == "долоо":
            return "долоон"
        if stem == "найм":
            return "найман"
        if stem == "ес":
            return "есөн"

        return stem  # fallback (should not happen for 1..9)

    # ---------------------------
    # Smart tokenize: join + split glued number words
    # ---------------------------
    def _smart_tokens(self, text: str):
        raw = (text or "").lower().split()
        out = []

        i = 0
        while i < len(raw):
            tok = raw[i]
            clean = re.sub(r"[^а-яөүё]", "", tok)
            if not clean:
                i += 1
                continue

            # ⭐ JOIN PATCH: applies to ALL 1..9 stems, not just "гурван"
            # pattern: "<digit> нь"  => connective-form, skip "нь"
            if i + 1 < len(raw):
                nxt = re.sub(r"[^а-яөүё]", "", raw[i + 1])
                if nxt == self._particle_n and clean in self._ones_stems:
                    out.append(self._to_connective_form(clean))
                    i += 2
                    continue

                # Also drop stray "нь" safely if it appears alone
                if clean == self._particle_n:
                    i += 1
                    continue

            # if already a known number word, keep
            if clean in self.values:
                out.append(clean)
                i += 1
                continue

            # if token contains markers, force-split FIRST
            pieces = self._split_by_markers(clean)
            if len(pieces) > 1:
                for piece in pieces:
                    if not piece:
                        continue
                    if piece == self._particle_n:
                        continue
                    if piece in self.values:
                        out.append(piece)
                        continue

                    # greedy split into known words
                    parts = []
                    k = 0
                    ok = True
                    while k < len(piece):
                        matched = None
                        for w in self._dict_words:
                            if piece.startswith(w, k):
                                matched = w
                                break
                        if not matched:
                            ok = False
                            break
                        parts.append(matched)
                        k += len(matched)

                    if ok and parts:
                        out.extend(parts)
                    else:
                        out.append(piece)

                i += 1
                continue

            # Original greedy split into known words
            parts = []
            k = 0
            ok = True
            while k < len(clean):
                matched = None
                for w in self._dict_words:
                    if clean.startswith(w, k):
                        matched = w
                        break
                if not matched:
                    ok = False
                    break
                parts.append(matched)
                k += len(matched)

            if ok and parts:
                out.extend(parts)
            else:
                out.append(clean)

            i += 1

        return out

    # ---------------------------
    # Amount math combine
    # ---------------------------
    def resolve_math(self, words):
        total = 0
        current = 0
        for w in words:
            val = self.values.get(w, 0)
            if val >= 100:
                current = (current if current > 0 else 1) * val
                if val >= 1000:
                    total += current
                    current = 0
            else:
                current += val
        return total + current

    # ---------------------------
    # Parse ONE group for account/iban
    # ---------------------------
    def _parse_group_basic(self, toks, i):
        n = len(toks)
        if i >= n:
            return None, i + 1

        w = toks[i]
        v = self.values.get(w, None)
        if v is None:
            return None, i + 1

        # hundreds: 5 зуу ( + 26 etc)
        if 0 <= v < 10 and i + 1 < n and toks[i + 1] in self._hundreds:
            base = v * 100
            j = i + 2

            if j < n and toks[j] in self._tens:
                base += self.values.get(toks[j], 0)
                j += 1
                if j < n and 0 <= self.values.get(toks[j], 99) < 10:
                    base += self.values.get(toks[j], 0)
                    j += 1

            return base, j

        # tens: 50 (+2)
        if w in self._tens:
            base = v
            j = i + 1
            if j < n and 0 <= self.values.get(toks[j], 99) < 10:
                base += self.values.get(toks[j], 0)
                j += 1
            return base, j

        # digit
        if 0 <= v < 10:
            return v, i + 1

        return v, i + 1

    # ---------------------------
    # Account/IBAN digits from groups (FIXED SCALE LOGIC)
    # ---------------------------
    def _digits_from_groups(self, fixed_list):
        toks = fixed_list
        out_groups = []
        i = 0
        n = len(toks)

        while i < n:
            # If a scale appears later, treat everything before it as multiplier phrase
            k = i
            while k < n and toks[k] not in self._scales_big:
                k += 1

            if k < n and toks[k] in self._scales_big and k > i:
                mult_phrase = toks[i:k]
                mult = self.resolve_math(mult_phrase)
                scale = self.values.get(toks[k], 0)
                mult = mult if mult > 0 else 1
                out_groups.append(mult * scale)
                i = k + 1
                continue

            # No scale ahead, parse basic groups sequentially
            val, j = self._parse_group_basic(toks, i)
            if val is not None:
                out_groups.append(val)
            i = max(j, i + 1)

        return "".join(str(g) for g in out_groups if g is not None)

    # ---------------------------
    # Public resolve
    # ---------------------------
    def resolve(self, text, mode="account"):
        # tokenize first to handle glued words + "нь" joining
        toks = self._smart_tokens(text)

        # then fix each token (Levenshtein to nearest vocab)
        fixed_list = [w for w in (self._fix(x) for x in toks) if w]
        fixed_text = " ".join(fixed_list)

        if not fixed_list:
            return "0", "N/A"

        if mode == "amount":
            return str(self.resolve_math(fixed_list)), fixed_text

        return self._digits_from_groups(fixed_list), fixed_text


# --- GLOBAL WRAPPERS (txn_router interface) ---
resolver = NumberResolver()


def resolve_amount_logic(text):
    v, f = resolver.resolve(text, mode="amount")
    return {"value": int(v or 0), "fixed_text": f}, 1.0


def resolve_account_logic(text):
    v, f = resolver.resolve(text, mode="account")
    return {"value": v, "fixed_text": f}, 1.0


def resolve_iban_logic(text):
    v, f = resolver.resolve(text, mode="account")
    return {"value": f"MN{v}", "fixed_text": f}, 1.0


def resolve_money(text):
    v, f = resolver.resolve(text, mode="amount")
    return int(v or 0), 0, 1.0
