import re
_tok = re.compile(r"[a-z0-9]+(?:[.\-@][a-z0-9]+)*")
STOP = set("the a an of to in is are for and or on at by with what how which when does do be it its as that this from per".split())
def tokens(s, drop_stop=True):
    t = _tok.findall(s.lower())
    return [w for w in t if w not in STOP] if drop_stop else t
