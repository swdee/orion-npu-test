# ---------------------------------------------------------------------
# Copyright 2025 Cix Technology Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------
import numpy as np

class strLabelConverter(object):
    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'

        self.dict = {}
        for i, char in enumerate(alphabet):
            self.dict[char] = i + 1
    def decode(self, t : np.ndarray, length : np.ndarray, raw : bool=False) -> str:
        """
        Decode the predicted sequence of indices into readable text.

        Args:
            t (np.ndarray): The predicted sequence of indices.
            length (np.ndarray): The length of the sequence.
            raw (bool, optional): If True, return the raw decoded text without removing duplicates or blanks. 
                                If False, return the cleaned text. Defaults to False.

        Returns:
            str: The decoded text.

        Raises:
            AssertionError: If the length of the sequence does not match the declared length.
        """
        if np.size(length) == 1:
            length = length.item()
            assert np.size(t) == length, "text with length: {} does not match declared length: {}".format(np.size(t), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            assert np.size(t) == np.sum(length), "texts with length: {} does not match declared length: {}".format(np.size(t), np.sum(length))
            texts = []
            index = 0
            for l in length:
                texts.append(
                    self.decode(
                        t[index:index + l], np.array([l]), raw=raw))
                index += l
            return texts