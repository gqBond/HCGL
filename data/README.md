# Dataset layout

Data files are not stored in this source release. Set `DATA_ROOT` to a directory containing the processed dataset folders.

Each dataset folder must contain these files:

```text
node.dat
link.dat
label.dat
label.dat.test
```

The existing loader also reads the schema encoded in these files. `meta.dat` and `info.dat` may be kept with the dataset for documentation, but the training code does not require them.

Expected target types and class counts are DBLP authors with 4 classes, IMDB movies with 5 classes, and Freebase books with 7 classes. Keep the preprocessing and train/test split used by the reported experiment. Cite and follow the license of each source dataset when publishing results or distributing data.
