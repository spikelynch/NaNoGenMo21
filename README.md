## Farbenlehre

Synaesthetic concrete poetry with source text Goethe's colour theory


## TODO

train an RNN on Goethe's Farbenlehre - DONE

modify the RNN sampler to allow lipogram generation - DONE

apply constraints by a function which maps x, y values on the page to "text colour" - DONE

map letters to RGB values according to my synaesthesia - DONE

rasterised lipogram - take an image and generate text such that each part of the printed text is "coloured" according to the RGB -> letter mapping

Try to make it work with headings and paragraph breaks


Retrain the RNN with some preprocessing to remove weird characters and improve the formatting - DONE

python ./farbenlehre.py -o book8 -n goethe2 -t 0.35 -c config.json

0.35 is a good balance between babble and not letting the "colours" through

