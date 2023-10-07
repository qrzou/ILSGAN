import click
import os
import json

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


@click.command()
@click.pass_context
@click.option('--source', help='Directory or archive name for input dataset', required=True, metavar='PATH')
def parse_labels(
    ctx: click.Context,
    source: str,
):
    classes = [d.name for d in os.scandir(source) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    instances = []
    directory = os.path.expanduser(source)
    def is_valid_file(x):
        return x.lower().endswith(IMG_EXTENSIONS)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)

    # rel path
    instances = [(os.path.relpath(path, directory), class_index) for path, class_index in instances]

    # save as json file
    with open(os.path.join(directory, 'dataset.json'), 'wt') as f:
        json.dump(dict(labels=instances), f)


if __name__ == '__main__':
    parse_labels()
