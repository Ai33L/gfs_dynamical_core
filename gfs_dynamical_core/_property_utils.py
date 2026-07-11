class GFSError(Exception):
    pass


def get_valid_properties(gfs_properties, prognostic_properties, property_type):
    return_dict = {}
    for name, properties in prognostic_properties.items():
        if name not in gfs_properties:
            return_dict[name] = properties
        elif "dims" in prognostic_properties.keys():
            extra_dims = set(prognostic_properties["dims"]).difference(
                ["*"] + list(gfs_properties[name]["dims"])
            )
            if len(extra_dims) != 0:
                raise GFSError(
                    "Cannot handle TendencyComponent with {} {} "
                    "that has extra dimensions {} not used by GFS".format(
                        property_type, name, extra_dims
                    )
                )
    return return_dict
