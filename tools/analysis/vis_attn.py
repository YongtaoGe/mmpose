import os
from argparse import ArgumentParser

from xtcocotools.coco import COCO

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)


def main():
    """Visualize the demo images.

    Require the json_file containing boxes.
    """
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for detection')
    parser.add_argument('pose_checkpoint', help='Checkpoint file')
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument(
        '--json-file',
        type=str,
        default='',
        help='Json file containing image info.')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='Root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')

    args = parser.parse_args()

    assert args.show or (args.out_img_root != '')

    coco = COCO(args.json_file)
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']

    img_keys = list(coco.imgs.keys())

    # optional
    return_heatmap = False

    # import pdb
    # pdb.set_trace()

    enc_self_attn_weights, dec_self_attn_weights, dec_cross_attn_weights, q2q_weights = [], [], [], []
    hooks = [
        # [[1, 2268, 256], [1, 2268, 256], [1, 2268, 256], [1, 2268, 256]]
        pose_model.keypoint_head.transformer.encoder.layers[-1].self_attn.register_forward_hook(
            lambda self, input, output: enc_self_attn_weights.append(output)
        ),

        # [17, 1, 256]
        pose_model.keypoint_head.transformer.decoder.layers[-1].self_attn.register_forward_hook(
            lambda self, input, output: dec_self_attn_weights.append(output[0])
        ),

        # [1, 17, 17]
        pose_model.keypoint_head.transformer.decoder.layers[-1].self_attn.register_forward_hook(
            lambda self, input, output: q2q_weights.append(output[1])
        ),

        #[[1, 17, 256], [1, 17, 256], [1, 17, 256], [1, 17, 256]]
        pose_model.keypoint_head.transformer.decoder.layers[-1].cross_attn.register_forward_hook(
            lambda self, input, output: dec_cross_attn_weights.append(output)
        ),

    ]

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None
    # process each image
    # for i in range(len(img_keys)):
    for i in range(1):
        # get bounding box annotations
        image_id = img_keys[i]
        image = coco.loadImgs(image_id)[0]
        image_name = os.path.join(args.img_root, image['file_name'])
        ann_ids = coco.getAnnIds(image_id)

        # make person bounding boxes
        person_results = []
        for ann_id in ann_ids:
            person = {}
            ann = coco.anns[ann_id]
            # bbox format is 'xywh'
            person['bbox'] = ann['bbox']
            person_results.append(person)

        # test a single image, with a list of bboxes
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            image_name,
            person_results,
            bbox_thr=None,
            format='xywh',
            dataset=dataset,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        if args.out_img_root == '':
            out_file = None
        else:
            os.makedirs(args.out_img_root, exist_ok=True)
            out_file = os.path.join(args.out_img_root, f'vis_{i}.jpg')

        vis_pose_result(
            pose_model,
            image_name,
            pose_results,
            dataset=dataset,
            kpt_score_thr=args.kpt_thr,
            show=args.show,
            out_file=out_file)


    for hook in hooks:
        hook.remove()

    # get the feature map shape
    # h, w = q2q_weights[0].shape[1:]
    #
    # fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
    # colors = COLORS * 100
    # for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
    #     ax = ax_i[0]
    #     ax.imshow(dec_attn_weights[0, idx].view(h, w))
    #     ax.axis('off')
    #     ax.set_title(f'query id: {idx.item()}')
    #     ax = ax_i[1]
    #     ax.imshow(im)
    #     ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
    #                                fill=False, color='blue', linewidth=3))
    #     ax.axis('off')
    #     ax.set_title(CLASSES[probas[idx].argmax()])
    # fig.tight_layout()


if __name__ == '__main__':
    main()


# # use lists to store the outputs via up-values
# conv_features, enc_attn_weights, dec_attn_weights = [], [], []
#
# hooks = [
#     model.backbone[-2].register_forward_hook(
#         lambda self, input, output: conv_features.append(output)
#     ),
#     model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
#         lambda self, input, output: enc_attn_weights.append(output[1])
#     ),
#     model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
#         lambda self, input, output: dec_attn_weights.append(output[1])
#     ),
# ]
#
# # propagate through the model
# outputs = model(img)
#
# for hook in hooks:
#     hook.remove()
#
# # don't need the list anymore
# conv_features = conv_features[0]
# enc_attn_weights = enc_attn_weights[0]
# dec_attn_weights = dec_attn_weights[0]
#
# # get the feature map shape
# h, w = conv_features['0'].tensors.shape[-2:]
#
# fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
# colors = COLORS * 100
# for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
#     ax = ax_i[0]
#     ax.imshow(dec_attn_weights[0, idx].view(h, w))
#     ax.axis('off')
#     ax.set_title(f'query id: {idx.item()}')
#     ax = ax_i[1]
#     ax.imshow(im)
#     ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
#                                fill=False, color='blue', linewidth=3))
#     ax.axis('off')
#     ax.set_title(CLASSES[probas[idx].argmax()])
# fig.tight_layout()
#
#
# # output of the CNN
# f_map = conv_features['0']
# print("Encoder attention:      ", enc_attn_weights[0].shape)
# print("Feature map:            ", f_map.tensors.shape)
#
#
# # downsampling factor for the CNN, is 32 for DETR and 16 for DETR DC5
# fact = 32
#
# # let's select 4 reference points for visualization
# idxs = [(200, 200), (280, 400), (200, 600), (440, 800),]
#
# # here we create the canvas
# fig = plt.figure(constrained_layout=True, figsize=(25 * 0.7, 8.5 * 0.7))
# # and we add one plot per reference point
# gs = fig.add_gridspec(2, 4)
# axs = [
#     fig.add_subplot(gs[0, 0]),
#     fig.add_subplot(gs[1, 0]),
#     fig.add_subplot(gs[0, -1]),
#     fig.add_subplot(gs[1, -1]),
# ]
#
# # for each one of the reference points, let's plot the self-attention
# # for that point
# for idx_o, ax in zip(idxs, axs):
#     idx = (idx_o[0] // fact, idx_o[1] // fact)
#     ax.imshow(sattn[..., idx[0], idx[1]], cmap='cividis', interpolation='nearest')
#     ax.axis('off')
#     ax.set_title(f'self-attention{idx_o}')
#
# # and now let's add the central image, with the reference points as red circles
# fcenter_ax = fig.add_subplot(gs[:, 1:-1])
# fcenter_ax.imshow(im)
# for (y, x) in idxs:
#     scale = im.height / img.shape[-2]
#     x = ((x // fact) + 0.5) * fact
#     y = ((y // fact) + 0.5) * fact
#     fcenter_ax.add_patch(plt.Circle((x * scale, y * scale), fact // 2, color='r'))
#     fcenter_ax.axis('off')