import git


def tag() -> str:
    return git.Repo(search_parent_directories=True).git.rev_parse("HEAD")
