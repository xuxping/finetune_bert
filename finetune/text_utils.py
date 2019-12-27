# -*- coding:utf-8 -*-

import re
import six
import unicodedata


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


REG_SCRIPT_TAG = r"<[\\s]*?script[^>]*?>[\\s\\S]*?<[\\s]*?/[\\s]*?script[\\s]*?>"  # 定义script的正则表达式
REG_STYLE_TAG = r"<[\\s]*?style[^>]*?>[\\s\\S]*?<[\\s]*?/[\\s]*?style[\\s]*?>"  # 定义style的正则表达式
REG_HTML_TAG = "<[^>]*?>"  # 定义HTML标签的正则表达式，也包含了html注释
SPACE_HTML = r"(&nbsp;)|(&#12288;)"  # Html空格占位字符
REG_XML_HTML = r"(&lt;)|(&gt;)|(&amp;)|(&quot;)|(&apos;)|(&copy;)"  # XML只有5个转义符
REG_SPECIAL_ESC = r"(&ldquo;)|(&rdquo;)|(&mdash;)|(&rarr;)|(&equiv;)|(&middot;)|(&#8220;)|(&#8221;)|(&#8212;)|(&lsquo;)|(&rsquo;)"  # 字符转义 “|”|—|→|≡|·|“|”|—


def remove_html(sentence):
    """移除html"""
    sentence = re.sub(REG_SCRIPT_TAG, '', sentence)
    sentence = re.sub(REG_STYLE_TAG, '', sentence)
    sentence = re.sub(REG_HTML_TAG, '', sentence)
    sentence = re.sub(SPACE_HTML, '', sentence)
    sentence = re.sub(REG_XML_HTML, '', sentence)
    sentence = re.sub(REG_SPECIAL_ESC, '', sentence)
    return sentence


def remove_strip(sentence):
    tokens = []

    for ch in sentence.strip():
        if is_whitespace(ch) or is_control(ch):
            continue
        tokens.append(ch)
    return ''.join(tokens)


def full_to_half(text):
    """全角转半角"""
    n = []
    for char in text:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        num = chr(num)
        n.append(num)
    return ''.join(n)


def is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False


def is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
