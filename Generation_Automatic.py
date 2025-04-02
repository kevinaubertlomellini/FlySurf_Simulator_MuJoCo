from scipy.sparse.csgraph import structural_rank
import math
import mujoco

def generate_xml(rows, cols, x_init, y_init, x_length, y_length, quad_positions, mass_points, mass_quads, str_stif, shear_stif, flex_stif, damp_point, damp_quad, delta, u_limits, file_path):
    """
    Generates XML for a grid of rows and columns with specified spacing and saves it to a file.

    Parameters:
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        x_init (float): Initial x-position.
        y_init (float): Initial y-position.
        x_length (float): Total length in x direction.
        y_length (float): Total length in y direction.
        quad_positions (list of list of int): List of [row, col] pairs for special elements.
        file_path (str): Path to save the XML file.

    Returns:
        None
    """
    # Order quad_positions
    quad_positions_ordered = sorted(quad_positions, key=lambda x: (x[1], x[0]))


    # Calculate spacing based on the total lengths and number of rows/columns
    x_spacing = x_length / (cols - 1)  # Adjusted for the correct number of divisions
    y_spacing = y_length / (rows - 1)  # Adjusted for the correct number of divisions

    xml_output = []

    # Add the Mujoco model header
    xml_output.append(f"""<mujoco model="Skydio X2">
  <compiler autolimits="true" assetdir="assets"/>
  <option timestep="{delta}"/>  <!-- Updated timestep -->
  <option gravity="0 0 -9.81"/>
  <default>
    <default class="ball">
        <geom size="0.005" mass="{mass_points}"/>
    </default>
    <default class="x2">
      <geom mass="0"/>
      <motor ctrlrange="0 13"/>
      <default class="visual">
        <geom group="2" type="mesh" contype="0" conaffinity="0"/>
      </default>
      <default class="collision">
        <geom group="3" type="box"/>
      </default>
      <site group="5"/>
    </default>
  </default>

  <asset>
    <texture type="2d" file="X2_lowpoly_texture_SpinningProps_1024.png"/>
    <material name="phong3SG" texture="X2_lowpoly_texture_SpinningProps_1024"/>
    <material name="invisible" rgba="0 0 0 0"/>
    <mesh class="x2" file="X2_lowpoly.obj" scale="0.0015 0.0015 0.0015"/>
  </asset>
  
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
""")

    element_counter = 1  # Start counting quads from 1
    actuator_output = []  # List to store the actuator entries
    horizontal_tendon_output = []  # List to store the actuator entries
    vertical_tendon_output = []  # List to store the actuator entries
    double_tendon_output = []  # List to store the actuator entries
    diagonal_tendon_output = []  # List to store the actuator entries

    # Loop through all grid positions
    for row in range(rows):
        for col in range(cols):
            x_pos = x_init + col * x_spacing
            y_pos = y_init + row * y_spacing

            # Check if the current position is in the quad_positions list
            # <joint type="free" damping="{damp_point}"/>
            if [row + 1, col + 1] in quad_positions_ordered:  # Positions are 1-indexed
                body_template = f"""
    <body name="quad_{element_counter}" pos="{x_pos:.4f} {y_pos:.4f} 0.05" childclass="x2">
        <joint type="free" damping="{damp_quad}"/>
        <geom material="phong3SG" mesh="X2_lowpoly" class="visual" quat="0 0 1 1"/>
        <geom class="collision" size=".009 .00405 .003" pos=".006 0 .003"/>
        <geom class="collision" size=".009 .00405 .003" pos=".006 0 .009"/>
        <geom class="collision" size=".0075 .00405 .003" pos="-.0105 0 .00975"/>
        <geom class="collision" size=".00345 .00255 .0015" pos="-.02055 .0012 .00975" quat="1 0 0 1"/>
        <geom name="point_mass_{element_counter}" type="sphere" size="0.012" pos="0 0 0" mass="{mass_quads}" material="invisible"/>
        <site name="ball_{row + 1}_{col + 1}" pos="0 0 0"/>
    </body>
    """
                # Add actuator for each quad
                actuator_output.append(f"""
    <motor class="x2" name="thrust{element_counter}_x" site="ball_{row + 1}_{col + 1}" gear="1 0 0 0 0 0" ctrlrange="{u_limits[0, 0]} {u_limits[0, 1]} "/>
    <motor class="x2" name="thrust{element_counter}_y" site="ball_{row + 1}_{col + 1}" gear="0 1 0 0 0 0" ctrlrange="{u_limits[1, 0]} {u_limits[1, 1]}"/>
    <motor class="x2" name="thrust{element_counter}_z" site="ball_{row + 1}_{col + 1}" gear="0 0 1 0 0 0" ctrlrange="{u_limits[2, 0]} {u_limits[2, 1]}"/>""")
                element_counter += 1
            else:
                body_template = f"""
    <body pos="{x_pos:.4f} {y_pos:.4f} 0.01">
        <joint type="free" damping="{damp_point}"/> 
        <geom class="ball"/>
        <site name="ball_{row + 1}_{col + 1}" pos="0 0 0"/>
    </body>
    """
            xml_output.append(body_template)
            if col+1 < cols:
                horizontal_tendon_output.append(f"""
    <spatial range="{x_spacing-0.01:.6f} {x_spacing+0.01:.6f}" limited="true"  width=".003" stiffness="{str_stif:.4f}">
	<site site="ball_{row + 1}_{col + 1}"/>
	<site site="ball_{row + 1}_{col + 2}"/>
    </spatial>""")
            if row + 1 < rows:
                vertical_tendon_output.append(f"""
    <spatial range="{y_spacing - 0.01:.6f} {y_spacing+0.01:.6f}" limited="true"  width=".003" stiffness="{str_stif:.4f}">
    <site site="ball_{row + 1}_{col + 1}"/>
    <site site="ball_{row + 2}_{col + 1}"/>
    </spatial>""")
            if col + 2 < cols:
                double_tendon_output.append(f"""
    <spatial range="{2*x_spacing - 0.02:.6f} {2*x_spacing+0.02:.6f}" limited="true"  width=".003" stiffness="{shear_stif:.4f}">
    <site site="ball_{row + 1}_{col + 1}"/>
    <site site="ball_{row + 1}_{col + 3}"/>
    </spatial>""")
            if row + 2 < rows:
                double_tendon_output.append(f"""
    <spatial range="{2*y_spacing - 0.02:.6f} {2*y_spacing + 0.02:.6f}" limited="true"  width=".003" stiffness="{shear_stif:.4f}">
    <site site="ball_{row + 1}_{col + 1}"/>
    <site site="ball_{row + 3}_{col + 1}"/>
    </spatial>""")
            if (col + 1 < cols) & (row + 1 < rows):
                diagonal_tendon_output.append(f"""
        <spatial range="{math.sqrt(y_spacing*y_spacing+x_spacing*x_spacing)- 0.012:.6f} {math.sqrt(y_spacing*y_spacing+x_spacing*x_spacing) + 0.012:.6f}" limited="true"  width=".003" stiffness="{flex_stif:.4f}">
        <site site="ball_{row + 1}_{col + 1}"/>
        <site site="ball_{row + 2}_{col + 2}"/>
        </spatial>""")
                diagonal_tendon_output.append(f"""
        <spatial range="{math.sqrt(y_spacing * y_spacing + x_spacing * x_spacing) - 0.012:.6f} {math.sqrt(y_spacing * y_spacing + x_spacing * x_spacing) + 0.012:.6f}" limited="true"  width=".003" stiffness="{flex_stif:.4f}">
        <site site="ball_{row + 2}_{col + 1}"/>
        <site site="ball_{row + 1}_{col + 2}"/>
        </spatial>""")

    # Close the worldbody tag
    xml_output.append("</worldbody>")

    # Add actuators dynamically after the worldbody
    xml_output.append("<actuator>")
    xml_output.extend(actuator_output)  # Add all actuator definitions
    xml_output.append("</actuator>")

    # Add tendons dynamically
    xml_output.append("<tendon>")
    xml_output.extend(horizontal_tendon_output)
    xml_output.extend(vertical_tendon_output)
    xml_output.extend(double_tendon_output)
    xml_output.extend(diagonal_tendon_output)
    xml_output.append("</tendon>")

    # Add the closing </mujoco> tag
    xml_output.append("</mujoco>")

    # Combine all generated XML snippets into a final string
    xml_content = "\n".join(xml_output)

    # Save to the specified file
    with open(file_path, "w") as file:
        file.write(xml_content)
    print(f"XML file saved to {file_path}")

    model = mujoco.MjModel.from_xml_path(file_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return [model, data]


def generate_xml2(rows, cols, x_init, y_init, x_length, y_length, quad_positions, mass_points, mass_quads, str_stif,
                 shear_stif, flex_stif, damp_point, damp_quad, delta, u_limits, max_l_str, max_l_shear, max_l_flex, file_path):
    """
    Generates XML for a grid of rows and columns with specified spacing and saves it to a file.

    Parameters:
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        x_init (float): Initial x-position.
        y_init (float): Initial y-position.
        x_length (float): Total length in x direction.
        y_length (float): Total length in y direction.
        quad_positions (list of list of int): List of [row, col] pairs for special elements.
        file_path (str): Path to save the XML file.

    Returns:
        None
    """
    # Order quad_positions
    quad_positions_ordered = sorted(quad_positions, key=lambda x: (x[1], x[0]))

    # Calculate spacing based on the total lengths and number of rows/columns
    x_spacing = x_length / (cols - 1)  # Adjusted for the correct number of divisions
    y_spacing = y_length / (rows - 1)  # Adjusted for the correct number of divisions

    xml_output = []

    # Add the Mujoco model header
    xml_output.append(f"""<mujoco model="Skydio X2">
  <compiler autolimits="true" assetdir="assets"/>
  <option timestep="{delta}"/>  <!-- Updated timestep -->
  <option gravity="0 0 -9.81"/>
  <default>
    <default class="ball">
        <geom size="0.005" mass="{mass_points}"/>
    </default>
    <default class="x2">
      <geom mass="0"/>
      <motor ctrlrange="0 13"/>
      <default class="visual">
        <geom group="2" type="mesh" contype="0" conaffinity="0"/>
      </default>
      <default class="collision">
        <geom group="3" type="box"/>
      </default>
      <site group="5"/>
    </default>
  </default>

  <asset>
    <texture type="2d" file="X2_lowpoly_texture_SpinningProps_1024.png"/>
    <material name="phong3SG" texture="X2_lowpoly_texture_SpinningProps_1024"/>
    <material name="invisible" rgba="0 0 0 0"/>
    <mesh class="x2" file="X2_lowpoly.obj" scale="0.0015 0.0015 0.0015"/>
  </asset>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>


  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
""")

    element_counter = 1  # Start counting quads from 1
    actuator_output = []  # List to store the actuator entries
    horizontal_tendon_output = []  # List to store the actuator entries
    vertical_tendon_output = []  # List to store the actuator entries
    double_tendon_output = []  # List to store the actuator entries
    diagonal_tendon_output = []  # List to store the actuator entries

    # Loop through all grid positions
    for row in range(rows):
        for col in range(cols):
            x_pos = x_init + col * x_spacing
            y_pos = y_init + row * y_spacing

            # Check if the current position is in the quad_positions list
            # <joint type="free" damping="{damp_point}"/>
            if [row + 1, col + 1] in quad_positions_ordered:  # Positions are 1-indexed
                body_template = f"""
    <body name="quad_{element_counter}" pos="{x_pos:.4f} {y_pos:.4f} 0.05" childclass="x2">
        <joint type="free" damping="{damp_quad}"/>
        <geom material="phong3SG" mesh="X2_lowpoly" class="visual" pos=".006 0 .02" quat="0 0 1 1"/>
        <geom class="collision" size=".009 .00405 .003" pos=".006 0 .003"/>
        <geom class="collision" size=".009 .00405 .003" pos=".006 0 .009"/>
        <geom class="collision" size=".0075 .00405 .003" pos="-.0105 0 .00975"/>
        <geom class="collision" size=".00345 .00255 .0015" pos="-.02055 .0012 .00975" quat="1 0 0 1"/>
        <geom name="point_mass_{element_counter}" type="sphere" size="0.012" pos="0 0 0" mass="{mass_quads}" material="invisible"/>
        <site name="ball_{row + 1}_{col + 1}" pos="0 0 0"/>
    </body>
    """
                # Add actuator for each quad
                actuator_output.append(f"""
    <motor class="x2" name="thrust{element_counter}_x" site="ball_{row + 1}_{col + 1}" gear="1 0 0 0 0 0" ctrlrange="{u_limits[0, 0]} {u_limits[0, 1]} "/>
    <motor class="x2" name="thrust{element_counter}_y" site="ball_{row + 1}_{col + 1}" gear="0 1 0 0 0 0" ctrlrange="{u_limits[1, 0]} {u_limits[1, 1]}"/>
    <motor class="x2" name="thrust{element_counter}_z" site="ball_{row + 1}_{col + 1}" gear="0 0 1 0 0 0" ctrlrange="{u_limits[2, 0]} {u_limits[2, 1]}"/>""")
                element_counter += 1
            else:
                body_template = f"""
    <body pos="{x_pos:.4f} {y_pos:.4f} 0.01">
        <joint type="free" damping="{damp_point}"/> 
        <geom class="ball"/>
        <site name="ball_{row + 1}_{col + 1}" pos="0 0 0"/>
    </body>
    """
            xml_output.append(body_template)
            if col + 1 < cols:
                horizontal_tendon_output.append(f"""
    <spatial range="{x_spacing - max_l_str:.6f} {x_spacing + max_l_str:.6f}" limited="true"  width=".003" stiffness="{str_stif:.4f}">
	<site site="ball_{row + 1}_{col + 1}"/>
	<site site="ball_{row + 1}_{col + 2}"/>
    </spatial>""")
            if row + 1 < rows:
                vertical_tendon_output.append(f"""
    <spatial range="{y_spacing - max_l_str:.6f} {y_spacing + max_l_str:.6f}" limited="true"  width=".003" stiffness="{str_stif:.4f}">
    <site site="ball_{row + 1}_{col + 1}"/>
    <site site="ball_{row + 2}_{col + 1}"/>
    </spatial>""")
            if col + 2 < cols:
                double_tendon_output.append(f"""
    <spatial range="{2 * x_spacing - max_l_shear:.6f} {2 * x_spacing + max_l_shear:.6f}" limited="true"  width=".003" stiffness="{shear_stif:.4f}" rgba="0.5 0.5 0.5 0">
    <site site="ball_{row + 1}_{col + 1}"/>
    <site site="ball_{row + 1}_{col + 3}"/>
    </spatial>""")
            if row + 2 < rows:
                double_tendon_output.append(f"""
    <spatial range="{2 * y_spacing - max_l_shear:.6f} {2 * y_spacing + max_l_shear:.6f}" limited="true"  width=".003" stiffness="{shear_stif:.4f}" rgba="0.5 0.5 0.5 0">
    <site site="ball_{row + 1}_{col + 1}"/>
    <site site="ball_{row + 3}_{col + 1}"/>
    </spatial>""")
            if (col + 1 < cols) & (row + 1 < rows):
                diagonal_tendon_output.append(f"""
        <spatial range="{math.sqrt(y_spacing * y_spacing + x_spacing * x_spacing) - max_l_flex:.6f} {math.sqrt(y_spacing * y_spacing + x_spacing * x_spacing) + max_l_flex:.6f}" limited="true"  width=".003" stiffness="{flex_stif:.4f}">
        <site site="ball_{row + 1}_{col + 1}"/>
        <site site="ball_{row + 2}_{col + 2}"/>
        </spatial>""")
                diagonal_tendon_output.append(f"""
        <spatial range="{math.sqrt(y_spacing * y_spacing + x_spacing * x_spacing) - max_l_flex:.6f} {math.sqrt(y_spacing * y_spacing + x_spacing * x_spacing) + max_l_flex:.6f}" limited="true"  width=".003" stiffness="{flex_stif:.4f}">
        <site site="ball_{row + 2}_{col + 1}"/>
        <site site="ball_{row + 1}_{col + 2}"/>
        </spatial>""")

    # Close the worldbody tag
    xml_output.append("</worldbody>")

    # Add actuators dynamically after the worldbody
    xml_output.append("<actuator>")
    xml_output.extend(actuator_output)  # Add all actuator definitions
    xml_output.append("</actuator>")

    # Add tendons dynamically
    xml_output.append("<tendon>")
    xml_output.extend(horizontal_tendon_output)
    xml_output.extend(vertical_tendon_output)
    xml_output.extend(double_tendon_output)
    xml_output.extend(diagonal_tendon_output)
    xml_output.append("</tendon>")

    # Add the closing </mujoco> tag
    xml_output.append("</mujoco>")

    # Combine all generated XML snippets into a final string
    xml_content = "\n".join(xml_output)

    # Save to the specified file
    with open(file_path, "w") as file:
        file.write(xml_content)
    print(f"XML file saved to {file_path}")

    model = mujoco.MjModel.from_xml_path(file_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return [model, data]
