"""Shared bits: path handling, atomic writes, a small worker pool, and the
folder / zip walking that btx.py, cls.py and ani.py are built on."""

import contextlib
import glob
import io
import multiprocessing
import os
import shutil
import sys
import threading
import zipfile
from collections import deque
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor, ThreadPoolExecutor

ARCHIVES = ('.zip', '.bpc')  # for these tools a .bpc is just a renamed zip


def cpus():
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def expand(paths):
    """Expand wildcards ourselves, windows and some android shells won't."""
    out = []
    for p in paths:
        if not os.path.exists(p) and any(c in p for c in '*?['):
            out += sorted(glob.glob(p, recursive=True)) or [p]
        else:
            out.append(p)
    return out


def bands(n, parts):
    """Split range(n) into at most `parts` (start, stop) pieces."""
    step = -(-n // max(1, min(parts, n)))
    return [(i, min(n, i + step)) for i in range(0, n, step)]


@contextlib.contextmanager
def atomic(path):
    """Yield a temp path next to `path`, move it over `path` if the block worked."""
    path = os.fspath(path)
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    tmp = os.path.join(folder, '.%s.%d.%d.tmp' % (os.path.basename(path), os.getpid(), threading.get_ident()))
    try:
        yield tmp
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def write(path, data):
    with atomic(path) as tmp, open(tmp, 'wb') as f:
        f.write(data)


class Pool:
    """Ordered parallel map. Processes if they work here, else threads, else a loop."""

    def __init__(self, jobs=0, procs=False, env=None):
        self.jobs = jobs if jobs > 0 else min(cpus(), 8)
        self.procs = procs
        self.env = env or {}
        self.ex = None
        self.started = self.jobs < 2

    def start(self):
        if self.started:
            return
        self.started = True
        os.environ.update(self.env)  # inherited by the workers
        if self.procs:
            try:
                ex = ProcessPoolExecutor(self.jobs, mp_context=multiprocessing.get_context('spawn'))
                ex.submit(abs, 1).result(timeout=120)  # make sure it really starts
                self.ex = ex
                return
            except Exception:
                pass
        self.ex = ThreadPoolExecutor(self.jobs)

    def submit(self, fn, task):
        if self.ex is not None:
            try:
                return self.ex.submit(fn, *task)
            except Exception:  # broken or shut down
                self.ex = None

    def map(self, fn, tasks, window=0):
        """Yield fn(*task) for each task, in order."""
        self.start()
        window = window or self.jobs * 3
        tasks, pending, done = iter(tasks), deque(), False
        while True:
            while not done and len(pending) < window:
                task = next(tasks, None)
                if task is None:
                    done = True
                else:
                    pending.append((task, self.submit(fn, task)))
            if not pending:
                return
            task, fut = pending.popleft()
            if fut is None:
                yield fn(*task)
                continue
            try:
                yield fut.result()
            except BrokenExecutor:
                self.ex = None
                yield fn(*task)

    def close(self):
        if self.ex is not None:
            self.ex.shutdown(wait=True)
            self.ex = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def infer(paths, exts):
    """The one extension from `exts` that all the given plain files share, else None."""
    if any(os.path.isdir(p) or p.lower().endswith(ARCHIVES) for p in paths):
        return None
    found = {os.path.splitext(p)[1].lower() for p in paths}
    return found.pop() if len(found) == 1 and found <= set(exts) else None


def walk(root, exts, skip=None):
    """Every file under root ending in one of `exts`, sorted, at any depth."""
    for cur, dirs, files in os.walk(root):
        dirs.sort()
        if skip:
            dirs[:] = [d for d in dirs if os.path.realpath(os.path.join(cur, d)) != skip]
        for name in sorted(files):
            if name.lower().endswith(exts):
                yield os.path.join(cur, name)


# --- converting -------------------------------------------------------------
# A rule is (src_ext, dst_ext, fn). fn(data, pool) returns bytes, or
# (bytes, note), and raises when the file can't be converted.

def call(fn, data, pool=None):
    """Run one conversion without ever raising: (ok, bytes or error, note)."""
    try:
        res = fn(data, pool)
    except Exception as e:
        return False, str(e) or type(e).__name__, ''
    return (True, *res) if isinstance(res, tuple) else (True, res, '')


def convert_file(fn, src, dst, pool=None):
    try:
        with open(src, 'rb') as f:
            data = f.read()
        ok, res, note = call(fn, data, pool)
        if ok:
            write(dst, res)
            res = ''
        return ok, res, note
    except OSError as e:
        return False, str(e), ''


def edit_zip(raw, rule, pool, prefix=''):
    """Convert the matching members of a zip, nested zips included.

    Returns (new zip bytes, log) with log = [(status, text)], status being
    ok, fail or skip. Members that fail are copied over untouched.
    """
    src, dst, fn = rule
    log, out = [], io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw)) as zin, zipfile.ZipFile(out, 'w') as zout:
        infos = zin.infolist()
        names = {i.filename for i in infos}
        kind = []
        for i in infos:
            low = i.filename.lower()
            if i.is_dir():
                kind.append('copy')
            elif low.endswith(ARCHIVES):
                kind.append('zip')
            elif low.endswith(src):
                skip = i.filename[:-len(src)] + dst in names
                kind.append('skip' if skip else 'convert')
            else:
                kind.append('copy')
        todo = ((fn, zin.read(i)) for i, k in zip(infos, kind) if k == 'convert')
        results = pool.map(call, todo)

        for info, k in zip(infos, kind):
            name, new_name, new_data = info.filename, info.filename, None
            if k == 'convert':
                ok, res, note = next(results)
                new_name = name[:-len(src)] + dst
                if ok:
                    new_data = res
                    log.append(('ok', f'{prefix}{name} -> {new_name}' + (f' ({note})' if note else '')))
                else:
                    new_name = name
                    log.append(('fail', f'{prefix}{name}: {res}'))
            elif k == 'skip':
                log.append(('skip', f'{prefix}{name}: {name[:-len(src)] + dst} already there'))
            elif k == 'zip':
                try:
                    inner, sub = edit_zip(zin.read(info), rule, pool, f'{prefix}{name}/')
                    log += sub
                    if any(s == 'ok' for s, _ in sub):
                        new_data = inner
                except zipfile.BadZipFile as e:
                    log.append(('fail', f'{prefix}{name}: {e}'))
            if new_data is None:
                new_data = zin.read(info)
            entry = zipfile.ZipInfo(new_name, info.date_time)
            entry.compress_type = info.compress_type
            entry.external_attr = info.external_attr
            entry.create_system = info.create_system
            entry.comment = info.comment
            zout.writestr(entry, new_data)
    return out.getvalue(), log


def convert_zip(rule, src, dst, pool):
    """Rewrite a zip (or renamed .bpc) on disk, keeping FILE.bak when done in place."""
    with open(src, 'rb') as f:
        raw = f.read()
    try:
        new, log = edit_zip(raw, rule, pool, src + '/')
    except zipfile.BadZipFile as e:
        hint = ' (encrypted? bpc.py decrypts it)' if src.lower().endswith('.bpc') else ''
        return [('fail', f'{src}: not a zip: {e}{hint}')]
    in_place = os.path.abspath(dst) == os.path.abspath(src)
    if any(s == 'ok' for s, _ in log) or not in_place:
        if in_place:
            shutil.copyfile(src, src + '.bak')
        write(dst, new)
    else:
        log.append(('skip', f'{src}: nothing to convert'))
    return log


def plan(paths, rule, out, archives):
    """Work out (files, zips, errors) from the command line paths."""
    src, dst, _ = rule
    single = len(paths) == 1 and os.path.isfile(paths[0])
    folder = None if single else out  # -o names one file only for a single input file
    skip = os.path.realpath(folder) if folder else None
    exts = (src,) + (ARCHIVES if archives else ())
    files, zips, errors = [], [], []

    def add(f, rel, base, explicit):
        low = f.lower()
        if archives and low.endswith(ARCHIVES):
            zips.append((f, out if single and out else (os.path.join(base, rel) if base else f)))
        elif low.endswith(src):
            if single and out:
                target = out
            elif base:
                target = os.path.join(base, os.path.splitext(rel)[0] + dst)
            else:
                target = os.path.splitext(f)[0] + dst
            files.append((f, target))
        elif explicit:
            errors.append(f'{f}: expected a {src} file')

    for p in paths:
        if os.path.isdir(p):
            root = os.path.normpath(p)
            base = folder and (os.path.join(folder, os.path.basename(root)) if len(paths) > 1 else folder)
            for f in walk(root, exts, skip):
                add(f, os.path.relpath(f, root), base, False)
        elif os.path.isfile(p):
            add(p, os.path.basename(p), folder, True)
        else:
            errors.append(f'{p}: no such file or directory')
    return files, zips, errors


def run(paths, rule, out=None, jobs=0, procs=False, env=None, archives=True):
    """Convert files, folders (any depth) and zips / .bpc given on the command line."""
    files, zips, errors = plan(expand(paths), rule, out, archives)
    fn = rule[2]
    count = {'ok': 0, 'fail': 0, 'skip': 0}

    def say(status, text):
        count[status] += 1
        print(text, file=sys.stderr if status == 'fail' else sys.stdout, flush=True)

    for e in errors:
        say('fail', 'error: ' + e)
    if not (files or zips):
        if not errors:
            print(f'nothing to convert: no {rule[0]} files found', file=sys.stderr)
        return 1

    with Pool(jobs, procs, env) as pool:
        if len(files) == 1 and not zips:  # one file: let the codec use the whole pool
            results = [convert_file(fn, *files[0], pool)]
        else:
            results = pool.map(convert_file, ((fn, s, d) for s, d in files))
        for (s, d), (ok, err, note) in zip(files, results):
            if ok:
                say('ok', f'{s} -> {d}' + (f' ({note})' if note else ''))
            else:
                say('fail', f'{s}: {err}')
        for s, d in zips:
            for status, text in convert_zip(rule, s, d, pool):
                say(status, text)

    if sum(count.values()) > 1:
        print('%(ok)d converted, %(fail)d failed, %(skip)d skipped' % count)
    return 1 if count['fail'] else 0
