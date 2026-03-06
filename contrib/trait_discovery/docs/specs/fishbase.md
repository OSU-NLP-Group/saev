I want to scrape the metadata from fishbase for all the species in FishVista.
Use `family` and `standardized_species` from all the *.csv files in /fs/ess/PAS2136/samuelstevens/datasets/fish-vista.
The README.md in that directory contains more info.

FishBase has links like this one:

https://www.fishbase.se/summary/Occidentarius_platypogon.html

On that page, we can find the link with text 'Summary page' that links to 

https://www.fishbase.se/webservice/summary/fb/showXML.php?identifier=FB-13479&ProviderDbase=03

Which is an XMl document.

We want to parse the XML document for the traits described in contrib/trait_discovery/data/fishbase.csv. We can use the same 0/1/?/"" format, where 0/1 indicate binary, ? indicates unknown from FishBase, and an empty string indicates missing. For real values like depth/pH, use the number instead of a 0/1.

There are many mirrors of fishbase:

> Mirrors : fishbase.org | fishbase.se | fishbase.de | fishbase.fr | fishbase.br | fishbase.au | fishbase.us | fishbase.ca 

Check for a crawl-delay on robots.txt. If it's an integer X, only make one request per X seconds to that particular mirror.

Use concurrent.futures + a threadpool, because we are likely IO bound, not CPU bound.
contrib/interactive_interp/classification/download/download_flowers.py and contrib/trait_discovery/scripts/format_butterflies.py have example of this sort of pattern.

This script should live in contrib/trait_discovery/scripts/scrape_fishbase.py, and use a Config class, use tyro, etc follow the same conventions as other scripts in this codebase.
