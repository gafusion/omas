import unittest
import omas
import os
import numpy as np
from omas.machine_mappings.d3d import ec_launcher_active_hardware
from omas.omas_machine import tunnel_mds

class TestD3D(unittest.TestCase):

    def test_d3d_ECH_machine_mapping(self):
        '''
        Test the machine mapping of the DIII-D ECH against the a .json file created with fetchGyrotron from the OMFIT TORAY module
        '''
        pristine_ods = omas.load_omas_json(os.path.join('tests', 'test_data', 'd3d_ECRH_machine_mapping.json'))
        test_ods = omas.ODS()
        tunnel_mds('d3d', 'RF')
        ec_launcher_active_hardware(test_ods, 170325)
        # test_ods.open('d3d', 170325)
        fields = ['launching_position.r', 'launching_position.z', 'launching_position.phi', 
                  'frequency.data', 'power_launched.data', 'mode.data', 'steering_angle_tor.data',
                  'steering_angle_pol.data']
        prefix = 'ec_launchers.launcher'
        for launch in pristine_ods[prefix]:
            for field in fields:
                full_field = prefix + f'[{launch}].' + field
                pristine_time = pristine_ods[prefix + f'[{launch}]'].time()[0] / 1.e3
                time_field = full_field.rsplit('.', 1)[0] + ".time"
                itime_test = np.argmin(np.abs(pristine_time - test_ods[time_field]))
                try:
                    self.assertAlmostEqual(pristine_ods[full_field][0], 
                            test_ods[full_field][itime_test])
                except AssertionError:
                    print("TEST FAILED FOR " + full_field)
                    print(f"Pristine time and value {pristine_time} {pristine_ods[full_field][0]}")
                    print(f"Test time and value {test_ods[time_field][itime_test]} {test_ods[full_field][itime_test]}")

if __name__ == '__main__':
    unittest.main()