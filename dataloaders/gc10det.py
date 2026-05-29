from utils import dataloader
import numpy as np


class GC10DETLoader():
    def __init__(self, args):
        self.args = args

    def _get_class_counts(self):
        labels = np.load(
            f"{self.args.data_dir}/{self.args.pretrained_model_name}/labels-{self.args.pretrained_model_name}.npy"
        )
        classes, counts = np.unique(labels, return_counts=True)
        if len(classes) != 10:
            raise ValueError(f"GC10-DET should have 10 classes, found {len(classes)} classes: {classes}")
        return dict(zip(classes.astype(int), counts.astype(int)))

    def _make_loader(self, base, increment):
        class_counts = self._get_class_counts()

        # Match the 50% labeled, 5-session protocol:
        # - Use 80% of the available training samples per class.
        # - Carry 20% of that amount from each previously encountered class.
        samples_per_class = int(min(class_counts.values()) * 0.8)
        if samples_per_class < 1:
            raise ValueError(f"Not enough GC10-DET samples per class: {class_counts}")

        num_labeled = base * samples_per_class
        num_novel_inc = samples_per_class
        num_known_inc = max(1, int(samples_per_class * 0.2))

        loader = dataloader.StrictPerClassIncrementalLoader(
            data_dir=self.args.data_dir,
            pretrained_model_name=self.args.pretrained_model_name,
            base=base,
            increment=increment,
            num_labeled=num_labeled,
            num_novel_inc=num_novel_inc,
            num_known_inc=num_known_inc,
        )

        train_loader = loader.train_dataloader()
        test_all_loader = loader.test_dataloader(mode='all')
        test_novel_loader = loader.test_dataloader(mode='novel')
        test_old_loader = loader.test_dataloader(mode='old')

        return train_loader, test_novel_loader, test_old_loader, test_all_loader

    def makeT5Loader(self):
        return self._make_loader(base=5, increment=1)

    def makeT10Loader(self):
        return self._make_loader(base=5, increment=1)

    def makeVinLoader(self):
        return self._make_loader(base=5, increment=5)
