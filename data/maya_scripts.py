import json
import maya.cmds as cmds


############################################################################
# List of vertex ids corresponding to landmarks
############################################################################


point_dict = {6: 989,
              7: 986,
              8: 1049,
              9: 1050,
              10: 1053,
              17: 201,
              18: 197,
              19: 327,
              20: 329,
              21: 420,
              22: 850,
              23: 764,
              24: 762,
              25: 646,
              26: 650,
              27: 15,
              28: 13,
              29: 10,
              30: 7,
              31: 308,
              32: 307,
              33: 3,
              34: 742,
              35: 743,
              36: 1101,
              37: 1096,
              38: 1092,
              39: 1089,
              40: 1086,
              41: 1106,
              42: 1081,
              43: 1078,
              44: 1074,
              45: 1069,
              46: 1064,
              47: 1084,
              48: 190,
              49: 107,
              50: 94,
              51: 21,
              52: 543,
              53: 556,
              54: 639,
              55: 571,
              56: 706,
              57: 28,
              58: 271,
              59: 122,
              60: 244,
              61: 1249,
              62: 1251,
              63: 1260,
              64: 678,
              65: 1234,
              66: 1223,
              67: 1225}

############################################################################
# Export Functions
############################################################################


# These point ids use a seperate mesh
special_list = [61, 62, 63, 65, 66, 77]

# Blendshape attribute names list
bs_list = ['BS.Mesh'] + ['BS.Mesh' + str(num) for num in range(1, 51)]


def get_points():
    # Get worldspace points, for default pose
    new_point_dict = {}
    for point in point_dict.keys():
        if point in special_list:
            new_point_dict[point] = cmds.pointPosition('trick_mesh.vtx[' + str(point_dict[point]) + ']')
        else:
            new_point_dict[point] = cmds.pointPosition('Neutral:Mesh.vtx[' + str(point_dict[point]) + ']')
    return new_point_dict


def get_points_offset(orig_points):
    # Get relative point offsets, for blendshapes
    new_point_dict = {}
    for point in point_dict.keys():
        if point in special_list:
            world_points = cmds.pointPosition('trick_mesh.vtx[' + str(point_dict[point]) + ']')
        else:
            world_points = cmds.pointPosition('Neutral:Mesh.vtx[' + str(point_dict[point]) + ']')

        new_point_dict[point] = [world_points[0] - orig_points[point][0],
                                 world_points[1] - orig_points[point][1],
                                 world_points[2] - orig_points[point][2]]
    return new_point_dict


############################################################################
# Gather data, export to JSON
############################################################################


def export_json():
    default_pose = get_points()
    bs_dict = {}

    for bs in bs_list:
        cmds.setAttr(bs, 1)
        bs_dict[bs] = get_points_offset(default_pose)
        cmds.setAttr(bs, 0)

    final_dict = {"default": default_pose, "blend_shapes": bs_dict}

    with open('bs_points_a.json', 'w') as f:
        json.dump(final_dict, f)
