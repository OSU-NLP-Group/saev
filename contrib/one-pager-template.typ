// One-pager project brief template.
// Copy this file into contrib/<project>/brief.typ and fill in the fields.

#let project-brief(
  title: none,
  abstract: none,
  motivation: none,
  goal: none,
  fig: none,
  data-and-methods: none,
  challenges: none,
  getting-started: none,
  collaborators: none,
  references: none,
  body,
) = {
  set page(paper: "us-letter", margin: (x: 0.75in, y: 0.75in), columns: 2)
  set par(justify: true)
  set text(size: 9pt)

  show link: set text(blue)
  show heading: set text(size: 10pt)

  // Title (spans both columns)
  place(
    top + center,
    float: true,
    scope: "parent",
    clearance: 1em,
  )[
    #align(center)[
      #text(16pt, weight: "bold")[#title]
    ]
  ]

  // Abstract
  if abstract != none {
    text(style: "italic")[#abstract]
  }

  // Motivation
  [== Motivation]
  motivation

  // Optional figure
  if fig != none {
    fig
  }

  // Goal
  [== Goal]
  goal

  // Data & Methods
  [== Data & Methods]
  data-and-methods

  // Challenges
  [== Challenges]
  challenges

  // Getting Started
  [== Getting Started]
  getting-started

  // Collaborators (at the end)
  if collaborators != none {
    [== Collaborators & Roles]
    collaborators
  }

  // References
  v(0.5em)
  line(length: 100%, stroke: 0.5pt)
  text(size: 8pt)[
    *SAE background:* #link("https://arxiv.org/abs/2511.17735")[Stevens et al. (2025a)] and #link("https://arxiv.org/abs/2502.06755")[Stevens et al. (2025b)]. #if references != none { references }
  ]
}
