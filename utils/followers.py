from __future__ import annotations

import os
from typing import Collection, Dict, Set


class Followers:
    def __init__(self, followers_dir, extension=".txt"):
        self._followers_dir = followers_dir
        if not os.path.exists(self._followers_dir):
            os.makedirs(self._followers_dir)
        self._extension = extension

    def keys(self):
        usernames_and_extensions = (
            os.path.splitext(filename) for filename in os.listdir(self._followers_dir)
        )
        return set(
            username
            for username, extension in usernames_and_extensions
            if extension == self._extension
        )

    def __len__(self):
        return len(self.keys())

    def _get_filename(self, username):
        return os.path.join(self._followers_dir, f"{username}{self._extension}")

    def __contains__(self, username):
        return os.path.exists(self._get_filename(username))

    def __getitem__(self, username):
        if not username in self:
            return set()
        with open(self._get_filename(username)) as file:
            return set(file.read().splitlines())

    def __setitem__(self, username, followers):
        with open(self._get_filename(username), "w") as file:
            file.write("\n".join(followers))

    def subset(self, usernames: Collection[str]) -> Dict[str, Set[str]]:
        return {name: self[name] for name in usernames}

    def clone(self) -> Followers:
        return Followers(self._followers_dir, self._extension)
