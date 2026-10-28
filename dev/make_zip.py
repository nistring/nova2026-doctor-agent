"""Build and check the submission ZIP from agent/.

  python dev/make_zip.py              # working tree agent/ -> dist/submission.zip (overwritten)
  python dev/make_zip.py --ref v8     # a tagged version -> dist/submission.zip
"""
import hashlib
import subprocess
import sys
import zipfile

from common import ROOT, export_ref

AGENT = ROOT / 'agent'
FILES = ['run.py', 'brain.py', 'nova_sdk.py', 'requirements.txt', 'knowledge/cp.json']
LIMITS = {'zipped': 50_000_000, 'unzipped': 500_000_000, 'entries': 5000}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def check():
    problems = []
    for name in FILES:
        path = AGENT / name
        if not path.is_file() or path.is_symlink():
            problems.append(f'missing or not a regular file: {name}')
        elif name.endswith(('.py', '.txt', '.json')):
            try:
                path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                problems.append(f'not UTF-8: {name}')
    if (AGENT / 'nova_sdk.py').exists() and sha(AGENT / 'nova_sdk.py') != sha(ROOT / 'nova-starter' / 'nova_sdk.py'):
        problems.append('nova_sdk.py differs from the starter package (must stay unmodified)')
    sources = [str(AGENT / n) for n in FILES if n.endswith('.py')]
    venv = ROOT / '.venv' / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')  # uv venv --python 3.12
    use_venv = venv.exists() and subprocess.run([str(venv), '-c', ''], capture_output=True).returncode == 0  # uv venvs break when their base Python moves
    python = [str(venv)] if use_venv else ['conda', 'run', '-n', 'nova312', 'python']
    compiled = subprocess.run([*python, '-c', 'import sys; assert sys.version_info[:2] == (3, 12), sys.version; '
                               '[compile(open(p, encoding="utf-8").read(), p, "exec") for p in sys.argv[1:]]',
                               *sources], capture_output=True, text=True, shell=sys.platform == 'win32' and not use_venv)
    if compiled.returncode:
        problems.append('Python 3.12 compile failed:\n' + compiled.stderr[-2000:])
    return problems


def main():
    global AGENT
    ref = sys.argv[sys.argv.index('--ref') + 1] if '--ref' in sys.argv else None
    if ref:
        AGENT = export_ref(ref)
    problems = check()
    if problems:
        sys.exit('\n'.join(['NOT BUILT:'] + problems))
    out = ROOT / 'dist' / 'submission.zip'
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(AGENT / name, name)
    with zipfile.ZipFile(out) as archive:
        infos = archive.infolist()
        unzipped = sum(i.file_size for i in infos)
    size = out.stat().st_size
    assert size <= LIMITS['zipped'] and unzipped <= LIMITS['unzipped'] and len(infos) <= LIMITS['entries']
    print(f'{out.relative_to(ROOT)}  {size:,} bytes zipped, {unzipped:,} unzipped, {len(infos)} files')
    print('root:', ', '.join(i.filename for i in infos))
    print('sha256:', sha(out))
    if ref:
        print('built from:', ref, '-', git('log', '-1', '--format=%s', ref))
        return
    print('built from:', git('describe', '--tags', '--always', '--dirty'), '-', git('log', '-1', '--format=%s'))
    if git('status', '--porcelain', '--', 'agent'):
        print('WARNING: agent/ has uncommitted changes; commit (and tag) before uploading so the ZIP maps to a version.')


if __name__ == '__main__':
    main()
