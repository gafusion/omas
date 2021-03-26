"""
W7-X additional structures and machine mappings
"""
import collections
import numpy as np

import omas


__all__ = []


from omas.omas_structure import add_extra_structures


# OMAS extra_structures
# Note: the name pf_active is of course very confusing and wrong. For now we charter this model
#   however
_stellarator_structures = {
    'pf_active': {
        "pf_active.circuit[:].supplies": {
            "coordinates": [
                "1...N"
            ],
            "data_type": "INT_1D",
            "documentation": "Description of the supplies in this circuit. The entries herein are references to the supplies under pf_active/supply. See pf_active/circuit(i1)/connections on how these are connected to the coils.",
            "full_path": "pf_active/circuit(i1)/supplies(:)",
            "lifecycle_status": "alpha",
            "type": "constant",
        },
        "pf_active.circuit[:].coils": {
            "coordinates": [
                "1...N"
            ],
            "data_type": "INT_1D",
            "documentation": "Description of the coils in this circuit. The entries herein are references to the coils under pf_active/coil. See pf_active/circuit(i1)/connections on how these are connected to the supplies.",
            "full_path": "pf_active/circuit(i1)/coils(:)",
            "lifecycle_status": "alpha",
            "type": "constant",
        },
    },
}


add_extra_structures(_stellarator_structures)


def add_circuits_from_coil_ids(
    ods,
    coil_ids,
    coil_types,
    supply_currents,
    coil_type_windings,
):
    """
    Note:
        It is assumed, that the coils are all regular in line
    """
    i = len(ods["pf_active.coil"])
    supply_offset = len(ods["pf_active.supply"])
    supplies = collections.OrderedDict()
    coil_indices = []
    for coil_id, coil_type in zip(coil_ids, coil_types):
        coil_indices.append(i)

        if coil_type in supplies:
            supply_id = supplies[coil_type]
        else:
            supply_id = supply_offset + len(supplies)

        currents = np.array([supply_currents[supply_id]])
        times = np.array([-1])

        ods[f"pf_active.coil.{i}.identifier"] = str(coil_id)
        ods[f"pf_active.coil.{i}.name"] = coil_type
        ods[f"pf_active.coil.{i}.current.data"] = currents
        ods[f"pf_active.coil.{i}.current.time"] = times
        ods[f"pf_active.coil.{i}.element.0.identifier"] = str(coil_id)
        ods[f"pf_active.coil.{i}.element.0.turns_with_sign"] = coil_type_windings[supply_id]
        i += 1

        ods[f"pf_active.supply.{supply_id}.current"]: currents
        ods[f"pf_active.supply.{supply_id}.itendifier"]: coil_type
        ods[f"pf_active.supply.{supply_id}.name"]: coil_type

    circuit_id = len(ods["pf_active.circuit"])

    ods[f"pf_active.circuit.{circuit_id}.supplies"] = np.array(list(supplies.values()))
    ods[f"pf_active.circuit.{circuit_id}.coils"] = np.array(coil_indices)
    ods[f"pf_active.circuit.{circuit_id}.current.data"] = currents


@omas.omas_machine.machine_mapping_function(__all__)
def active_coils(ods, coil_version='ideal', stellarator_symmetric_ic_coils=True):
    r"""
    Loads W7-X coil hardware geometry

    :param ods: ODS instance
    """

    if coil_version == 'ideal':
        # Non Planar Coils
        add_circuits_from_coil_ids(
            ods,
            list(range(160, 210)),
            [f"{coil_version.upper()} Non Planar Coil {(id_ % 5) + 1}" for id_ in range(160, 210)],
            [13.2e3] * 5,
            [108] * 5,
        )

        # Planar Coils
        add_circuits_from_coil_ids(
            ods,
            list(range(210, 230)),
            [f"{coil_version.upper()} Planar Coil {['A', 'B'][id_ % 2]}" for id_ in range(210, 230)],
            [0] * 5,
            [36] * 5,
        )

        # Island Control Coils
        add_circuits_from_coil_ids(
            ods,
            list(range(230, 240)),
            [f"{coil_version.upper()} Island Control Coil {(id_ % 2) + 1}" for id_ in range(230, 240)],
            [0] * 2,
            [8] * 2,
        )

        # Island Control Coils
        add_circuits_from_coil_ids(
            ods,
            [350, 241, 351, 352, 353],
            [
                f"{coil_version.upper()} Trim Coil A",
                f"{coil_version.upper()} Trim Coil B",
                f"{coil_version.upper()} Trim Coil A",
                f"{coil_version.upper()} Trim Coil A",
                f"{coil_version.upper()} Trim Coil A",
            ],
            [0] * 5,
            [46, 72, 46, 46, 46],
        )


if __name__ == '__main__':
    omas.ODS()
    omas.omas_machine.run_machine_mapping_functions(__all__, globals(), locals())
    # update_mapping()
