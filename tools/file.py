import os
import pickle
import shutil
import sys

from tqdm import tqdm


def create_all_dirs(path):
    if "." in path.split("/")[-1]:
        dirs = os.path.dirname(path)
    else:
        dirs = path
    os.makedirs(dirs, exist_ok=True)


def get_files_paths(dataroot, extensions, data_type='normal'):
    # '''
    # '''
    '''
    get file path list, support lmdb or normal files     lmdb暂时还没测试，不一定能用
    :param dataroot: 要输出文件路径的目录
    :param data_type: normal file  或者 lmdb
    :param extensions: 文件的扩展名
    :return:
    '''
    assert isinstance(extensions, list) or isinstance(extensions,
                                                      str), "extensions必须str或list, e.g: “png” or ['.jpg', '.JPG']"

    def _get_paths_from_lmdb(dataroot):
        '''get image path list from lmdb meta info'''
        meta_info = pickle.load(open(os.path.join(dataroot, 'meta_info.pkl'), 'rb'))
        paths = meta_info['keys']
        sizes = meta_info['resolution']
        if len(sizes) == 1:
            sizes = sizes * len(paths)
        return paths, sizes

    def _get_paths_from_normal(path, extensions):
        '''get file path list from file folder'''
        assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)

        def is_extension_file(filename, extensions):
            EXTENSIONS = []
            if isinstance(extensions, str):
                EXTENSIONS.append(extensions)
            else:
                EXTENSIONS = extensions
            return any(filename.endswith(extension) for extension in EXTENSIONS)

        flies = []
        for dirpath, _, fnames in sorted(os.walk(path)):
            for fname in sorted(fnames):
                if is_extension_file(fname, extensions):
                    img_path = os.path.join(dirpath, fname)
                    flies.append(img_path)

        if not flies:
            print('{:s} has no valid file'.format(path))
        return flies

    paths, sizes = None, None
    if data_type == 'lmdb':
        if dataroot is not None:
            paths, sizes = _get_paths_from_lmdb(dataroot)
        return paths, sizes
    elif data_type == 'normal':
        if dataroot is not None:
            paths = sorted(_get_paths_from_normal(dataroot, extensions))
        return paths
    else:
        raise NotImplementedError('data_type [{:s}] is not recognized.'.format(data_type))


def get_file_name(path, with_extension=True):
    if with_extension:
        return os.path.basename(path)
    else:
        return os.path.basename(path).split('.')[0]


def separate_files_by_txt(file_path, txt_path, save_path):
    '''
    按照txt分类文件, txt文件有两列：第一列文件名，第二列类别

    txt looks like:
                            000001.jpg 0
                            000002.jpg 0
                            000003.jpg 0
    :param file_path: 文件目录
    :param txt_path:  txt路径

    e.g.:     separate_files_by_txt("../../datasets/CelebA/img_align_celeba", "../../datasets/CelebA/list_eval_partition.txt",
                          "../../datasets/CelebA")
    '''

    with open(txt_path, "r") as f:
        for line in tqdm(f.readlines()):
            line_l = line[:-1].split(" ")
            try:
                create_all_dirs(os.path.join(save_path, line_l[1]))
                shutil.copy(os.path.join(file_path, line_l[0]), os.path.join(save_path, line_l[1], line_l[0]))
            except IOError as e:
                print("Unable to copy file. %s" % e)
            except:
                print("Unexpected error:", sys.exc_info())


def separate_files_by_parts(file_path, save_path, extensions, part_num):
    '''
    把文件等分成几分
    :param file_path:
    :param save_path:
    :return:
    '''

    def div_list(ls, n):
        result = []
        cut = int(len(ls) / n)
        if cut == 0:
            ls = [[x] for x in ls]
            none_array = [[] for i in range(0, n - len(ls))]
            return ls + none_array
        for i in range(0, n - 1):
            result.append(ls[cut * i:cut * (1 + i)])
        result.append(ls[cut * (n - 1):len(ls)])
        return result

    files = get_files_paths(file_path, extensions)
    assert len(files) % part_num == 0, print("文件不能被等分")
    files_list = div_list(files, part_num)

    for idx, part in tqdm(enumerate(files_list)):
        create_all_dirs(os.path.join(save_path, 'part_' + str(idx)))
        for file in part:
            try:
                shutil.copy(file, os.path.join(save_path, 'part_' + str(idx), get_file_name(file)))
            except IOError as e:
                print("Unable to copy file. %s" % e)
            except:
                print("Unexpected error:", sys.exc_info())

