import os
import subprocess

from acl.snapshot import has_commit, read_python_files


def git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def test_reads_only_regular_py_files_at_commit(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    git(work, "init", "-q")
    (work / "pkg").mkdir()
    (work / "pkg" / "a.py").write_text("x = 1\n")
    (work / "README.md").write_text("hi\n")
    os.symlink("pkg/a.py", work / "link.py")
    git(work, "add", ".")
    git(work, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "one")
    first = git(work, "rev-parse", "HEAD")
    (work / "pkg" / "a.py").write_text("x = 2\n")
    git(work, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "two")

    bare = tmp_path / "m.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(work), str(bare)], check=True)
    assert has_commit(bare, first)
    assert not has_commit(bare, "0" * 40)
    assert read_python_files(bare, first) == {"pkg/a.py": "x = 1\n"}
