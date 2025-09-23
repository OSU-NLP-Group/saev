# CONTRIBUTING

## 1. Welcome & Scope

`saev` is research code.
PRs that fix bugs, add datasets, or improve docs are welcome.
Large architectural rewrites: please open a discussion first.

## 2. TL;DR

Install [uv](https://docs.astral.sh/uv/).
Clone this repository, then from the root directory:

```sh
uv run python -m saev --help
```

You also need [yek](https://github.com/bodo-run/yek) and [lychee](https://github.com/lycheeverse/lychee) for generating docs.

If you want to do any of the web interface work, you need [elm](https://guide.elm-lang.org/install/elm.html), [elm-format](https://github.com/avh4/elm-format/releases/latest) and [tailwindcss](https://github.com/tailwindlabs/tailwindcss/releases/latest).

## 3. Testing & Linting

`justfile` contains commands for testing and linting.

`just lint` will format and lint.
`just test` will format, lint and test, then report coverage.

To run just one test, run `uv run python -m pytest src/saev -k TESTNAME`.

## 4. PR Checklist

1. Run `just test`.
2. Check that there are no regressions. Unless you are certain tests are not needed, the coverage % should either stay the same or increase.
3. Run `just docs`.
4. Fix any missing doc links.

## 5. Research Reproducibility Notes

If you add a new neural network or other hard-to-unit-test bit of code, it should either be a trivial change or it should come with an experiment demonstrating that it works.

This means some links to WandB, or a small report in markdown in the repo itself.
For example, if you wanted to add a new activation function from a recent paper, you should train a small sweep using the current baseline, demonstrate some qualitative or quantitative results, and then run the same sweep with your minimal change, and demonstrate some improvement (speed, quality, loss, etc).
Document this in a markdown report (in `src/saev/nn` for a new activation function) and include it in the docs.

Neural networks are hard. It's okay.

## 8. Code of Conduct & License Footnotes

Be polite, kind and assume good intent.
