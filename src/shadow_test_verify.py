
import os
import mujoco
import numpy as np
from pathlib import Path

def run_shadow_test_local():
    project_root = Path(os.getcwd())
    print(f"Executing Shadow Robot Test in {project_root}...")
    
    # 1. Mock XML if not exists for the test
    xml_content = """
    <mujoco>
        <worldbody>
            <body name="pelvis">
                <joint type="free"/>
                <geom size="0.1" type="sphere"/>
                <body name="thigh">
                    <joint name="hip" type="ball"/>
                    <geom size="0.05 0.1" type="capsule"/>
                </body>
            </body>
        </worldbody>
    </mujoco>
    """
    
    # Ensure Dynamics compartment has the template
    template_path = project_root / "Compartments" / "Dynamics" / "humanoid_template.xml"
    if not template_path.exists():
        with open(template_path, "w") as f:
            f.write(xml_content)
            
    model = mujoco.MjModel.from_xml_string(xml_content)
    data = mujoco.MjData(model)
    
    # 2. Apply 50Nm torque
    test_torque = 50.0
    data.qfrc_applied[6] = test_torque
    
    # 3. Step physics
    mujoco.mj_step(model, data)
    
    # 4. Acceleration check
    accel = data.qacc[6]
    print(f"Applied Torque: {test_torque}Nm")
    print(f"Recorded Acceleration: {accel:.6f} rad/s^2")
    
    # 5. SSD Record
    ledger_path = project_root / "Compartments" / "Ledger" / "shadow_test.atoms.npz"
    np.savez_compressed(ledger_path, torque=test_torque, accel=accel)
    
    if ledger_path.exists():
        print(f"SUCCESS: Physical data persisted to local SSD -> {ledger_path}")
    else:
        print("FAILURE: SSD write failed.")

if __name__ == "__main__":
    run_shadow_test_local()
