_Colourless Light_ is an attempt to use my own grapheme-colour synaesthesia to modulate the output of a recurrent neural network so that the distribution of letters on the page forms a pattern. I've always had strong one-way associations between letters, numbers and colours - the glyphs evoke colours, but the colours don't evoke glyphs - and I have a taste for lipograms, texts which avoid certain letters, the most famous of which is Georges Perec's novel _La Disparition_, translated into English as _A Void_ by Gilbert Adair, and one of the things which strikes me about them is that they are different colour, in the synaesthetic sense, from a normal text. The letter E, for me, is a strong, bright red, and most of the contribution to a word's synaesthetic hue comes from its vowels, so when I read Perec's novel, the vowel colours from A (white), I (transparent), O (black) and U (deep purple), make the text seem different: darker, less saturated, closer to the blue end of the spectrum.

I've been tinkering with lipogrammatic RNNs for a while now - for a recent experiment see [gravidum cor](https://bots.mikelynch.org/gravidum_cor/), a Mastodon bot which generates a simulation of Burton's Anatomy of Melancholy without E. What I wanted to do for this year's NaNoGenMo was make an RNN generate text with a different mixture of letters on different parts of the page. I had no idea how this would look to my synaesthetic sense - whether the invisible colours of the letters would translate when arranged spatially in this way.

At first I had the idea that one could take an image and map its RGB components onto the appropriate letters and make a "picture", but my early experiments with rendering two different constrained outputs into a checkerboard proved that patterns on a scale of ten or so letters weren't really detectable. I settled on using two large-scale patterns: dividing the page into two halves along a diagonal, and placing a circle of one colour on a background of another. The colour combinations are chosen by an algorithm which moves through every combination of the eight basic colours in turn, ensuring that each combination of two colours is rendered with each of the geometrical patterns.

I found that my initial idea of rendering different colours by suppressing letters didn't work very well. If I suppressed all letters other than those which look red to me (E, J, S and 3) the resulting output was too constrained to make sense. I tried another strategy, of trying to make red text by suppressing the letters and numbers which are complementary to it, but this didn't produce results which looked red. Many samples of experiments in this vein can be found [in the outputs/suppresssion/ folder in the repository](https://github.com/spikelynch/NaNoGenMo21/tree/main/output/suppression).

So, rather than suppressing letters, I modified the RNN to increase the probability of the relevant letter set. Using [this tutorial](https://www.tensorflow.org/text/tutorials/text_generation) as a starting point, I modified the part of the one-step sampler which masks out error characters to boost the probability of a particular set of glyphs: here is an example of text with E, J, S and 3 (and their lower-case counterparts) boosted:

```
n so seen these exerspessing selesses, so the eyes seems even sees even seen
even especially sees jessessed sees in justages seems even some eleveness.
See sees. "everything sees these experiments essees in semi-transparents
meetes seems sees in some sees essential to each other seems to see sees these
seconds essentially seess, these spectra messessed seen these exerspessing
substances see yellow
```

The patterns were implemented by defining a function which takes an x, y pair defining the location on the page, and returns a set of character weights which will generate the correct colour for that point. The book is rendered in a monospaced font as a shortcut: theoretically it would be possible to calculate the location of each letter in a proportional-width font as it was rendered, but this would have involved too much intertwining of the text generation and the typesetting. I quite liked the default monospace font chosen by Pandoc, with its faint aroma of computer journals and concrete poetry anthologies from the seventies, but it doesn't include enough Unicode to render the Greek letters which the RNN occasionally generates, so I used DejaVu Sans Mono instead.

The RNN was trained on [the Gutenberg edition Goethe's Theory of Colour in Charles Lock Eastlake's translation](https://www.gutenberg.org/cache/epub/50572/pg50572-images.html), both because it generates an appropriately colour-obsessed text, and because of Goethe's importance in the history of the philosophy of colour. Goethe was the defender of a notion of colour based on phenomenology and affect, in defiance of the scientific consensus following Newton. The colours of synaesthesia, whether they arise from childhood associations or some cross-wiring of sensory modalities on the individual brain, fall more on Goethe's side of this debate.

Finally, although my initial idea that a fine-grained colour map could be used to produce a detailed image via a black and white text didn't work, I can report that the pages of _Colourless Light_ do seem to me to be coloured, in the curious indirect way in which grapheme synaesthesia works. It's never as if the characters themselves are coloured: the colour seems to be on some other plane. But looking at these pages is as close as I've felt to being able to see my own synaesthesia. It's even more solipsistic an exercise than NaNoGenMo usually is: you will be able to notice the different letter distributions, but even if you also have grapheme synaesthesia, you won't see the same colours as I do. Of course, you can edit [the config file](https://github.com/spikelynch/NaNoGenMo21/tree/main/config.json), and generate your own.

### Instructions

You'll need to install TensorFlow and PyYAML, and put the [trained checkpoint files](https://etc.mikelynch.org/nanogenmo2021/) in the directory `./trained_checkpoints/goethe`.

The script to generate the novel as markdown is:

```
% python ./farbenlehre.py -o myversion -n goethe -c config.json
```

The PDF version was generated with Pandoc:

```
pandoc --pdf-engine=xelatex -o output/myversion/colourless_light.pdf \
    output/myversion/colourless_light.md
```


### Configuration

```
	"latex": {
		"title": "Colourless Light",
		"subtitle": "NaNoGenMo 2021",
		"author": "Mike Lynch",
		"geometry": "margin=3.6cm",
		"linkcolor": "blue",
		"monofont": "DejaVu Sans Mono"

	},
	"appendix": "README.md",
	"output": "colourless_light.md",
	"temperature": 0.4,
	"page": {
		"width": 80,
		"height": 40
	},
	"order": [
		"black", 
		"blue", "green", "yellow",
		"orange", "red", "purple", "gray",
		"white"
		],
	"strength": 7,
	"steepness": 3,
	"colours": {
		"red": "ejs3",
		"purple": "dpru89",
		"blue": "bkmnw4",
		"green": "t2",
		"yellow": "cl7",
		"orange": "fghxq5",
		"white": "aiy1",
		"gray": "zv6",
		"black": "o0"
	}
```

#### latex

A collection of Pandoc variables which are converted to YAML and added to the top of the Markdown file - these set the title, subtitle, author, margins, font and link colour.

#### appendix

Name of the file (this one!) which gets added to the text as the Appendix

#### output

Filename for the Markdown output

#### temperature

Temperature parameter for the RNN

#### page

Page geometry with width and height

#### order

Order of the colours fed to the algorithm which generates the sequence of colour pairs

#### strength

Weighting given to the letters which are being promoted

#### steepness

Steepness of the gradient between colour regions on the page. Higher values give a sharper transition.

#### colours

A mapping from colour names to sets of letters and numbers. The script will automatically include upper and lower case versions of letters.

