# New Project Structure

saev is structured like [big_vision](https://github.com/google-research/big_vision), Google's ViT codebase.
To get the most use out of saev, you should not use it as a requirement in your project; rather, you should build inside of the source code of saev.
This is a guide to that process.

**TL;DR:**

1. Fork saev.
2. Clone your fork.
3. Create a new directory in `contrib/`.
4. Update both `src/saev` and your new contrib directory as necessary.
5. (Hopefully) publish.
5. If your changes to `src/saev` are broadly useful and not overly restrictive, open a PR with your changes to `src/saev`.

I am currently applying SAEs to audio of [birdsong](https://en.wikipedia.org/wiki/Bird_vocalization), so this is how I'll develop it.

First, fork and clone saev.
Do this however you want, but [GitHub has a guide on it](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo).

Second, you probably want to store code related to your project in this repo.
Make a new directory in `contrib/`. I'm calling my new subproject "birdsong."

```
[I] samuelstevens@host ~/p/saev (main)> tree -L 1 contrib/
contrib/
├── birdsong
├── interactive_interp
└── trait_discovery
```

Use `uv` to make a new package inside your new project:

```
[I] samuelstevens@host ~/p/s/c/birdsong (main)> uv init --package .
Adding `birdsong` as member of workspace `~/projects/saev`
Initialized project `birdsong` at `~/projects/saev/contrib/birdsong`
```

Now you have some additional files.

```
[I] samuelstevens@ascend-login02 ~/p/s/c/birdsong (main)> tree
.
├── pyproject.toml
├── README.md
└── src
    └── birdsong
        └── __init__.py
```

Now I can write scripts and source code for birdsong-specific stuff in here.
I'll probably add a notebook for looking at instances of birdsongs before and after using SAEs to identify patterns under a new `birdsong/notebooks` directory, and will add `birdsong/logbook.md` to store ongoing TODO items, and so on.

To train SAEs on audio files, I'll need to add a new dataset type to save activations.
In order to do this, I'll edit `src/saev/data/datasets.py`.

I'll also need to add another model to the dataset, one that expects audio files.
Since I don't think that DINOv3, OpenCLIP, or the other existing model families will be suitable, I'll need to add a new model family.
Again, this will need to go somewhere in `src/saev/data`.

If I'm smart about it, these changes will be nice and non-destructive, and other users of saev can benefit from them.
After I publish some results, to share this code with others, I'll open a PR from my fork/branch to main with the new datasets/models.
But I won't open a PR with `birdsong` because that's specific to me, rather than to the library.[^pr]

[^pr]: Technically, `birdsong` will be in saev because I'm a sort of privileged user because I'm the main developer. But other folks probably want their project-specific code attached to their GitHub page, rather than OSU-NLP's.
