from os import path as osp
import pandas as pd
import numpy as np
import colorsys
from PIL import Image
from PIL import ImageOps
from collections import defaultdict

# processing color grids

def parse_utterances(utterances):
    return [(u["sender"], u["contents"], u["time"]) for u in utterances]


def parse_state_and_action(state_data, action_data):
    state = state_data["state"]
    action = action_data["action"]

    objs = state["objs"]
    condition = state["condition"]["name"]

    target = state["target"]
    speaker_order = state["speakerOrder"]
    listener_order = state["listenerOrder"]

    listener_clicked = action["lClicked"]
    listener_selection = listener_order[action["lClicked"]]
    success = listener_selection == target

    return (
        objs,
        condition,
        target,
        speaker_order,
        listener_order,
        listener_clicked,
        success,
    )


def parse_round(r):
    rn = r["roundNum"]
    gameid = r["gameid"]
    events = r["events"]

    events_dict = defaultdict(list)
    for e in events:
        events_dict[e["eventType"]].append(e)

    if len(events_dict["utterance"]) >= 1:
        parsed_utterances = parse_utterances(events_dict["utterance"])
        parsed_utterances = sorted(parsed_utterances, key=lambda x: x[2])
    else:
        parsed_utterances = []

    n_states = len(events_dict["state"])
    n_actions = len(events_dict["action"])

    assert n_states == n_actions

    if n_states == 1:

        events_dict["state"] = events_dict["state"][0]
        events_dict["action"] = events_dict["action"][0]

        objs, condition, target, speaker_order, listener_order, listener_clicked, success = parse_state_and_action(
            events_dict["state"], events_dict["action"])

    elif n_states == 0:
        objs = condition = target = speaker_order = listener_order = listener_clicked = success = None

    round_dict = {
        "gameid": gameid,
        "roundNum": rn,
        "utterances": parsed_utterances,
        "objs": objs,
        "condition": condition,
        "target": target,
        "speaker_order": speaker_order,
        "listener_order": listener_order,
        "listener_clicked": listener_clicked,
        "success": success,
    }

    return round_dict


# building item images


def hsl_to_rgb(h, s, l):
    h = h / 360 if h > 1 else h
    s = s / 100 if s > 1 else s
    l = l / 100 if l > 1 else l
    return colorsys.hls_to_rgb(h, l, s)


def hls_to_rgb(h, l, s):
    h = h / 360 if h > 1 else h
    l = l / 100 if l > 1 else l
    s = s / 100 if s > 1 else s
    return colorsys.hls_to_rgb(h, l, s)


def pil_format(rgb_tuple):
    rgb_array = np.array(rgb_tuple)
    rgb_array = (rgb_array * 255).astype(np.uint8).reshape(1, 1, -1)
    return rgb_array


def pad_img(img, size, color):
    return ImageOps.pad(img, size, color=color)


def surround_with_padding(img, padding=10, color=(255, 255, 255)):
    size = max(img.size)
    new_size = int(size + padding)
    out_img = pad_img(
        pad_img(img, (new_size, size), color), (new_size, new_size), color
    )
    return out_img


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def build_images(
    objs,
    order,
    patch_size=300,
    patch_padding=50,
    patch_pad_color=(255, 255, 255),
    grid_padding=50,
    grid_pad_color=(255, 255, 255),
):

    images = []

    for idx in order:
        # select next grid (following listener order)
        obj = objs[idx]
        # patches to rgb
        rgbs = [pil_format(hsl_to_rgb(*shape["color"])) for shape in obj["shapes"]]
        # PIL Images -> Color patches
        imgs = [Image.fromarray(rgb).resize((patch_size, patch_size)) for rgb in rgbs]
        padded_imgs = [
            surround_with_padding(img, round(patch_padding / 2), patch_pad_color)
            for img in imgs
        ]
        grid = image_grid(padded_imgs, 3, 3)
        images.append(grid)

    # concatenate grids

    return images

def build_grid_image(
    objs,
    order,
    patch_size=100,
    patch_padding=10,
    grid_padding=50,
    patch_pad_color=(255, 255, 255),
    grid_pad_color=(255, 255, 255),
):

    obj_grids = []

    for idx in order:
        # select next grid (following listener order)
        obj = objs[idx]
        # patches to rgb
        rgbs = [pil_format(hsl_to_rgb(*shape["color"])) for shape in obj["shapes"]]
        # PIL Images -> Color patches
        imgs = [Image.fromarray(rgb).resize((patch_size, patch_size)) for rgb in rgbs]
        # pad Patches, make 3x3 grid
        padded_imgs = [
            surround_with_padding(img, round(patch_padding / 2), patch_pad_color)
            for img in imgs
        ]
        grid = image_grid(padded_imgs, 3, 3)
        grid = surround_with_padding(grid, round(patch_padding / 2), patch_pad_color)
        # pad grid
        padded_grid = surround_with_padding(grid, grid_padding, color=grid_pad_color)

        obj_grids.append(padded_grid)

    # concatenate grids
    item_img = image_grid(obj_grids, 1, 3)

    return item_img


def build_grid_image_with_target_highlight(
    objs,
    order,
    patch_size=100,
    patch_padding=10,
    grid_padding=50,
    patch_pad_color=(255, 255, 255),
    grid_pad_color=(255, 255, 255),
    target_idx=None,
    target_pad_color=(0, 255, 0),
):

    obj_grids = []

    for idx in order:
        # select next grid (following listener order)
        obj = objs[idx]
        # patches to rgb
        rgbs = [pil_format(hsl_to_rgb(*shape["color"])) for shape in obj["shapes"]]
        # PIL Images -> Color patches
        imgs = [Image.fromarray(rgb).resize((patch_size, patch_size)) for rgb in rgbs]
        # pad Patches, make 3x3 grid
        padded_imgs = [
            surround_with_padding(img, round(patch_padding / 2), patch_pad_color)
            for img in imgs
        ]
        grid = image_grid(padded_imgs, 3, 3)

        if target_idx is not None and idx == target_idx:
            # highlight target
            grid = surround_with_padding(grid, round(patch_padding / 2), (target_pad_color))
        else:
            # default pad color
            grid = surround_with_padding(grid, round(patch_padding / 2), (patch_pad_color))

        # pad grid
        padded_grid = surround_with_padding(grid, grid_padding, color=grid_pad_color)

        obj_grids.append(padded_grid)

    # concatenate grids
    item_img = image_grid(obj_grids, 1, 3)

    return item_img


def build_patch_image(
    objs, order, patch_size=300, patch_padding=50, patch_pad_color=(255, 255, 255)
):

    patches = []
    for idx in order:
        hls_tuple = objs[idx]
        rgb = hls_to_rgb(*hls_tuple)
        img = Image.fromarray(pil_format(rgb)).resize((patch_size, patch_size))
        padded_img = surround_with_padding(img, patch_padding, patch_pad_color)
        patches.append(padded_img)

    # concatenate grids
    item_img = image_grid(patches, 1, 3)

    return item_img


def build_patch_image_with_target_highlight(
    objs,
    order,
    patch_size=300,
    patch_padding=50,
    patch_pad_color=(255, 255, 255),
    highlight_target=False,
    target_pad_color=(0, 255, 0),
):

    patches = []
    for idx in order:
        hls_tuple = objs[idx]
        rgb = hls_to_rgb(*hls_tuple)
        img = Image.fromarray(pil_format(rgb)).resize((patch_size, patch_size))

        if highlight_target and idx == 0:
            # highlight target
            padded_img = surround_with_padding(img, patch_padding, target_pad_color)
        else:
            # default padding color
            padded_img = surround_with_padding(img, patch_padding, patch_pad_color)
        patches.append(padded_img)

    # concatenate grids
    item_img = image_grid(patches, 1, 3)

    return item_img


ordered_gameids = [
    '4147-9d9582f2-28d0-431f-b6ba-35fd1be76619',
    '3514-6cfc8b8b-11b8-45a4-b882-53d3d59f207f',
    '1813-732cf2c6-d39b-4f8a-a5fe-c9a3c90d62e6',
    '0067-e57d7967-3791-4fac-844c-4a8f8aa6e38a',
    '1844-2a44d690-09ff-49c9-8091-82d3b9b89a85',
    '1009-21dc2f3d-18a2-455e-abc3-d7defbbb52e5',
    '1204-b62356ce-b952-48b3-a19a-92ca76854fd8',
    '7115-f5686dd6-2f2d-4e05-bd29-9aef492f056c',
    '8762-e53d5964-95c9-4178-b810-c583f99f3ccc',
    '9028-13f162e4-c741-48c7-a43f-bef11c54d69e',
    '8702-4794397c-acb7-4766-bc65-df5df5ad9ef1',
    '5261-f18d8ffc-a679-493d-9588-204ed448fddc',
    '9149-79167050-b7dd-481c-8985-a3e3da8a5fa5',
    '0784-a452055c-d66a-49d3-8b3c-c78f288869bd',
    '8134-8d65c2ad-2bae-4daf-9eba-2409c0cb4fd0',
    '6165-643f06f5-c772-4d31-ab7e-fc8b47126e03',
    '5612-5c88c9c6-08f9-4a85-9d55-f13bee5d9f26',
    '3763-d036c1f6-aeab-4a7a-a55c-4bfe7ca1463a',
    '9059-918d13d3-8dc3-4197-a59c-b4bcf80ee3b9',
    '1826-5ecde216-48a0-4af2-a53a-586a56a3ec54',
    '4156-101a6c08-35cf-4074-b00c-50cd58e9e5ad',
    '4183-8fd72d7b-65da-4edf-8769-e3b78734409c',
    '1071-1610f674-2a91-4633-86a7-a25fa6ac492e',
    '1906-40c4b2ef-171b-4f0b-b70e-fb0955631406',
    '5935-5acbd802-5826-4bcb-95d7-e5eeb563eeae',
    '4640-094098b3-2a8d-4518-b3e4-84d553e0b4b6',
    '5046-ed82846a-25d1-4105-836b-e9fa800949df',
    '8190-0f2f16da-9551-40a7-ab02-f1282d283649',
    '3025-ab77e2ae-479f-4f18-9f5a-bda7b7723f58',
    '2290-d25c3561-59f0-4819-9413-1fc24bf769ce',
    '2363-3ecb42e3-5ebf-478e-95a9-cb17af0455bf',
    '7273-e3b8be5c-1dc3-41be-add2-3a196a365227',
    '8169-9b8e5970-7823-4042-9bf9-93ea8a54b6a5',
    '5869-b05f29c2-0e96-4594-8b1a-6c292bbe4e60',
    '2694-2b626eb1-587f-492f-89b6-5ab0d357e578',
    '1579-224e0c88-c357-4052-81ae-e7856a45cfc2',
    '9335-e74f76e9-037b-4200-b6fe-da45c3f544bd',
    '9418-0432c528-7af2-433c-a3b9-5186b5d76e75',
    '6803-954fecee-67f3-47df-bd60-ddd5cdec07ea',
    '1256-f4864ba1-6b67-4cff-bb6e-3ada066e9e42',
    '9882-f6bd3009-3bfe-445b-8d99-02f672aa634c',
    '1207-2dca376b-dd8f-4ce9-b6bd-56c62ce26a91',
    '8897-ce8224bc-de34-4ffd-883c-0ecf0991ebc7',
    '7661-3f287857-01c5-4cc2-97e7-c53cf16669a5',
    '9343-8287585f-0c67-4e71-a8fc-c0b840487ee1',
    '9521-b2e5b5a2-49e7-4668-b273-c2537fbaeb40',
    '9276-7e1adc54-c703-4630-8631-2f1b57ac05c3',
    '2069-8261d590-bb07-4ddb-bf66-c7d232ba337f',
    '8435-66cacb66-b9ff-4d42-86b8-dd7b01834215',
    '6287-05c4b490-4e6a-4849-b2c3-b788fab7f84b',
    '6677-6bfea9dd-7c54-41f2-98f1-6efa4e779d8e',
    '2195-6b0ab1e6-3ac9-4a24-ad41-cf931ecb95cb',
    '7086-79b4b0a8-9d2c-4f84-920d-d788aaaf54d3',
    '0465-7a50dfff-a4ea-4beb-b736-05cec15136f1',
    '7020-223d401a-dc24-452c-8abd-bb8b8ea95004',
    '6608-eb22a117-71d9-4ec1-b8e7-2c118a799090',
    '2512-c48d4369-49f5-4500-af1f-5e3c759473d7',
    '6719-2700e8eb-c2ee-46b3-bcbb-543695fe2f3c',
    '5948-6739fbf7-ad1a-4a61-bcbb-03cd77eb7b66',
    '7396-5f1763e1-18c0-4544-9a33-d03be5418d97',
    '7298-85dab9e7-fe1c-4c3d-9358-f08c67d6edf4',
    '7258-7f97f685-2951-4f40-b6ee-de1020600a8d',
    '5764-eb14aac8-a047-4406-b622-50502cd5423f',
    '7430-634c64c7-d57b-4131-94b9-b00cf56e24d5',
    '4054-bc4c37eb-5528-42ea-a4d8-2123371fbdd7',
    '1856-9f11ca07-82a2-401e-9b4a-96ef592df43d',
    '4011-91ffe3c6-22fb-46a6-9455-4216bc23a30a',
    '4332-b8654eaf-1f17-478a-aded-a5a03e54b064',
    '8454-2e0a9860-ef78-4574-b9aa-8591c6d5541f',
    '9593-58e327f4-87be-4bce-9468-e9086f5d190b',
    '6353-7200c6ee-f38f-4907-b467-695fe11c3046',
    '5402-8813ac48-67df-460d-89d0-5f2537ab1a23',
    '2982-6442cb70-1654-4cfc-b4d2-dac8c02ec939',
    '4291-4ba5fa41-7d12-43d7-8908-1930b504910e',
    '5388-9f6ebac3-a77c-49ad-af4f-f1ca25ae863e',
    '3916-97de76bb-2bfc-47e9-93c9-fa3e4062c905',
    '7093-0451d886-5aab-47d0-8a5b-155e2d6d7e2b',
    '7799-8cb6d719-9f1e-4b63-91d7-826e3674f528',
    '5626-77df9de8-51f1-4187-b7da-540a54583944',
    '8383-e6e5d8d8-f814-4f89-9a2b-1b7ff41475ec',
    '8568-0ab2082e-8dfe-4690-b5ac-a0c2c9e5ce5b',
    '8989-7c5ab155-cb9b-444d-bcee-eeb2ab691004',
    '3260-908c65bf-0b84-4a51-b4a1-f5e48837de0a',
    '9349-2c5efa5e-d39b-48fd-84ea-530d757748fb',
    '3439-e55035d9-a653-47c7-8889-2e428183eedb',
    '7392-d73401ae-77c9-438b-96e6-bebe9f8a3b65',
    '9280-79fd84a7-7847-44e1-b60f-4b1d9395ce49',
    '6912-69032d64-b515-49e5-bbc1-67fce3a9356e',
    '0467-701ce84b-aa3c-4ad4-bb34-ffb872036a5b',
    '3263-c906efc3-d3b0-4ad3-98fe-6aa15bb05d22',
    '0106-d3c57ab1-95ed-4684-8631-48aa4e964a07',
    '9765-4d41f70b-6dcc-423c-97a5-94e9b1f30717',
    '6631-91f4fea6-3d11-452e-b674-1329b97ccc27',
    '5775-c100e8d2-fcaa-4249-8745-1ae09dab7ac0',
    '9471-06065311-a1b1-4641-8420-6f52730b8141',
    '0701-a3455492-d837-415e-819b-0d81d19efe53',
    '5232-8e082d35-5c55-42d1-9ce5-53af30d6c8dd',
    '9597-b2286185-8375-4c3a-81a1-842be70a921e',
    '1365-f75aa281-4887-4663-9826-b639f2440a6c',
    '5260-906c0c0b-a4cd-4ac0-920d-d00c6c947345',
    '7216-460382d4-3246-45d0-a241-751eac1380bc',
    '9345-777fb152-a0aa-4c84-8343-b9ce82de277b',
    '9955-1cbd5506-e782-422f-84e7-70366d4f805c',
    '9478-37fd786c-a6c0-4e5c-a448-6cb7cda4ba22',
    '8934-4373ceb5-7742-492d-ae04-4312bbcdc736',
    '7213-9e079b0f-2183-4a7c-aed7-ea6d4681ba4e',
    '7099-e2f5be8f-9248-4fa5-bd96-ebd3ce977d32',
    '4007-0feb7c2d-df2c-438d-8f2c-0d5438e7e6fb',
    '5670-ddb8e63c-03d3-41df-a6e0-79f2fb3e82ab',
    '8508-190b62fb-5776-469f-8adc-95a66c271fc9',
    '0256-42f8883f-fbe6-4b2e-bcfa-60eaa92ca24d',
    '7484-6ed1fee6-2d0a-4f45-86f2-096b64448b96',
    '9193-e80b9de9-7035-4132-a05d-786e925f9e1c',
    '6242-49130cee-e2e7-4f18-bb62-991472d41eee',
    '0881-f78c506e-05cf-483f-b592-b5df7667a5df',
    '4100-62d8ad3a-446c-416f-9f44-f95be73ed13b',
    '2706-bb37466b-b7dc-4567-942f-b1c3713069e4',
    '0982-4119815d-2c27-4519-b320-a364d7160a07',
    '1483-c427534b-3b97-486d-9e93-17ad1e10dcb6',
    '2566-8683291b-d29a-4ee4-9f73-4daf232fecdf',
    '6381-e772be69-e74d-4df6-8e82-7fcdcb2b7f97',
    '6327-c8039526-d991-43c6-8688-8888a32d3901',
    '3793-6f103aa0-9478-4dcf-b162-93e5a56d2720',
    '5541-ffb402cf-9f49-4fb4-9089-18141546ce1f',
    '8169-bb232933-05f1-4fe7-b57c-1e75d88dbe5f',
    '8818-2ae9907b-935d-4a1c-880a-7866ebea2cff',
    '5578-244673da-3dcc-436f-a742-d754537ed590',
    '6650-f081af26-5622-41af-b9e7-7ccfb787d269',
    '2095-86d2ba1e-81d0-41fe-84e3-f7b97ceba778',
    '9987-b57203f6-9f9a-4111-a058-f1779bbf86ff',
    '4845-dba9d58b-4bbe-45e2-8dd3-bace3186eb07',
    '4585-9a26cb9b-0937-46df-ab3b-ab58c2587a30',
    '2589-050e0f91-7dba-48c6-b219-c1d2a9b4eb39',
    '3335-f60a68e2-a21a-4b5d-9a4f-f9f2dd673077',
    '3537-3fbe46b1-5345-48ec-a657-60edffe607cf',
    '8947-20d9cd70-d0da-4406-82c9-dacd37727c93',
    '2807-73a9cdbb-5dbc-4a00-aa1c-157d3bb6d1aa',
    '8112-d96bdbf4-aec9-4b5a-812b-2a525055ebe4',
    '7713-4a76a6cd-d877-4da1-b5a1-5ae73ef79c48',
    '5808-36b08e75-7c16-4f3c-b65c-642d990c726f',
    '6723-45763cd6-1ad2-4d33-9864-c52f0b4e23f7',
    '8727-45f86bb2-5254-4329-860e-26afb17e1631',
    '3460-f4f9fa1e-55ba-490b-b446-355c7ca88c93',
    '2004-ab63189a-39bc-4540-a4e2-671949f6319a',
    '5370-ffc73313-d8a7-4a26-97e1-55bd32f4149c',
    '8638-bc8928d9-13aa-4df1-a83f-6b5897ff32bb',
    '4772-3a8b74ae-de5b-40ad-9b58-641a375181c7',
    '6742-30dc1dc7-31a4-4f0d-9648-7e93930249e1',
    '9641-9ff24c0b-f085-4b5d-b174-20186bf4a696',
    '6113-38555c00-9ed9-463a-ab14-04cdfb258539',
    '1039-b3c8ef96-bc52-4e52-a814-0389a926c898',
    '9674-736edb2a-2d95-4b49-8f8f-2f3f5d80c57b',
    '7189-f0d37330-ab59-4af3-833d-275aa257f6f0',
    '8941-c761c5d2-c134-4132-b16e-2d0b5d696e27',
    '2518-654cdf37-9bd8-4e00-bef3-7c5667e79c41',
    '7567-1dac5777-d852-47b8-930b-d7f01824404b',
    '2006-30a4b375-6559-417a-bd27-4162d9961186',
    '2845-5eb20470-8193-41e7-a326-ff6f23812693',
    '3683-8bcf4877-22e7-47dc-ba28-36b28ac2144c',
    '9619-634592c1-e6b0-4944-82f6-6e4711fa44df',
    '9344-80e5cfd9-609d-4c4d-a737-626a54de5af7',
    '2362-28cbc35e-2542-495b-8053-8ab762f7dad9',
    '9177-17516e8e-aefd-48f4-adfa-712bf98dfa80',
    '0662-16810e9b-6397-4bb9-9b30-faacb5fa0af0',
    '1963-570d3d28-9850-4738-884e-fc3ecb8832a3',
    '0261-db92856e-7a0b-49b4-b867-2f255736c651',
    '1792-263e0f86-5cc1-45c5-9f6c-7bb4a24a9cee',
    '4875-b0bf8007-ee17-4937-b9a7-29326048fb83',
    '6316-3b7a5b8a-752f-4bb6-9d38-d2a7030ac4cb',
    '2774-76ce6304-17f8-4cba-8580-deeef1161e9b',
    '0781-6d39940b-f8b8-4b3a-addb-a7445d06f2c3',
    '1463-5c2c95c7-2401-41ff-b34c-98e93536f94c',
    '1818-27bcf60d-7100-4a3e-8fee-59ad6b6051f0',
    '4134-ce8c7b41-296d-407f-af4e-e01ae44a150c',
    '5102-c056bf1c-6bca-486e-af9f-7c8ba26d575d',
    '9073-736207f4-765c-4868-b463-fe4a7bd3b763',
    '7254-d6cc4dd7-61a8-4aec-b827-a775d1172836',
    '2090-ebb58aea-8cb6-40b1-9283-6af7fd389f5a',
    '1557-242d3520-9331-46b0-921e-a4345faee5f9',
    '0356-2f7908fb-0dcc-4e43-81d9-412057901ca8',
    '8332-62d15a69-ebda-4b60-a759-5b338129bed6',
    '7169-96fbd601-23ed-435b-880a-939d4785f5e9',
    '4447-68443758-056e-4aae-b0a4-1776a4f20133',
    '1885-0596178c-0a8e-47e4-8c99-d70c66b06fd9',
    '4583-6c31b5e0-a28a-462a-86de-0005576071e6',
    '0133-f20abda3-ae08-4f8c-9b0a-b1566a83e141',
    '6015-97d353e8-0235-43ca-8f3c-110324222e4a',
    '7372-ee5a3c9e-332f-45d5-ad5e-797364d85d36',
    '3030-33d15b15-58c2-49c4-9ab5-09f8d7379aee',
    '0052-70ed266e-1bd0-4c34-ae83-0db5bc103a26',
    '5207-e36d6a16-30b0-47b2-a969-743b0f0ffe1d',
    '0626-f0128351-e82d-49c1-bd5a-00f561372e4e',
    '1488-1f65da2d-a2d3-4832-8ebc-048baa59cabc',
    '5704-eb65850a-01b8-43b7-ab26-937928c2f46c',
    '3980-0934c1fd-6cde-4d3d-b6eb-8af16e6bb98f',
    '0553-3313762d-5e5b-44a3-8faa-686f1873746d',
    '6914-2c7e7061-57cb-43ad-9a1f-1549e260c1ff',
    '6068-c8135454-2875-41d9-918c-60c34fc9460c'
]

gameid_order = {gameid:i for gameid,i in zip(ordered_gameids, range(len(ordered_gameids)))}
