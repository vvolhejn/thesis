import datetime
import os


# This function is also in `util` but that module imports TensorFlow, which is slow
def get_today_string():
    """0331, 0611 etc."""
    today = datetime.date.today()
    date_s = today.strftime("%m%d")
    return date_s


def add_distinguishing_suffix(base_dir, name, ignore_empty_dirs=True):
    """
    Choose a dir name that doesn't exist yet by trying to append '-1', '-2' and so on.
    n.b.: `base_dir` is not added to the beginning of the returned path.
    """
    n = 0
    while True:
        candidate = name
        if n > 0:
            candidate += f"-{n}"

        path = os.path.expanduser(os.path.join(base_dir, candidate))
        if not os.path.exists(path):
            return candidate
        elif ignore_empty_dirs and os.path.isdir(path) and os.listdir(path) == []:
            # If the directory exists but is empty, this probably means an earlier
            # attempt at submission crashed. Reuse the directory in this case.
            return candidate
        else:
            n += 1
