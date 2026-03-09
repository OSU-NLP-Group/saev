# Logbook

> (Human-written notes)

# 03/08/2026

I think we need to consider some different factors for each latent and write a score for each latent and each image in a parquet file.

- Have we used parquet in this project before?
- How big would that file be? n latents x n images x n scores x 4 bytes?

Remember, we have mimic pair tasks, where each task has a certain number of images.
So not all images are going to be used.
And most scores are only relevant in a particular mimic pair context.

What defines a good latent?

1. Highly discriminative for predicting melpomene vs erato for a given subspecies pair
2. High precision and recall. When it's active, it's class A. When it's off, it's class B. But you can define this for some threshold (pick a weight + bias, and it's a linear probe).
3. Spatially coherent

# 03/09/2026

1pwpq6ue, lativitta vs malleti dorsal:

- 1335 is hammer and nails (known)
- 3388 is hammer and nails (known)
- 6712 is hammer and nails (known)
- 961 is hammer and nails (known)
- 8732 is the tips of the nails, not sure what the difference is (unknown?)
- 10250 is the ray shape (known)

1pwpq6ue, notabilis vs plesseni dorsal:

- 5374 is the orange patch on forewings
- 12423 is the orange patch on forewings

kmlavddy, cyrbia vs cythera dorsal:

- 12823 is the black/white fringe (known)
- 10054 is the orange patch being pointy/not pointy (unknown?)

kmlavddy, cyrbia vs cythera ventral:

- 7583 is a shape? color? hard to tell, but it's discriminative
- 18834 is something about the forewing's orange/white stripe

kmlavddy, notabilis vs plesseni ventral:

- 17733 is the four dots

kmlavddy, venus vs vulcanus ventral:

- 21290 is something about the forewing white patch border (unknown?)


Raw Notes from Aly on MEE:

- Power of SAEs -> intro (Owen)
- Methods - High level overview, background, nothing nitty-gritty
  - details in supplemental
  - discussion of metrics in results -> talk about intricacies
- scorecard kind of style (if else, scoring of when something is useful with table and color codes) -> indicated scorecard is very appealing
  - see https://www.researchgate.net/publication/349830423_Towards_monitoring_ecosystem_integrity_within_the_Post-2020_Global_Biodiversity_Framework, final figure
- maybe tex, maybe word doc
- discussion : scorecard to broader ecology
methods -> results -> discussion

no graphical abstract/conceptual figure (only needed if it makes it to peer review)

-> no algorithms, put it in supplemental
  -> graphical flowchart instead of an algorithm

word count (7k) for MEE

