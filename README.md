
# Rhino3MFPaintSupport

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<a href="https://www.spherene.ch">
  <img src="https://spherene.ch/wp-content/uploads/2022/06/Spherene-logo-black.svg" alt="Spherene" width="200" style="filter: invert(1);" />
</a>

## Overview

This repository contains a CPython script for **Rhino 8** designed to generate **painted supports** on a 3D mesh and export the result as a `.3mf` file suitable for 3D printing slicers (Tested with Bambu Studio, Prisa Slicer and Orca Slicer).

Supports can be generated based on two main criteria (which can be combined for better paint supports):

1. **Proximity to Selected Points:** Mesh faces near user-selected Rhino Point objects are marked for support. Includes optional neighborhood expansion to enlarge the supported regions.
2. **Vertex Color & Face Orientation:** Mesh faces are marked for support if they have predominantly **green** vertex colors (indicating potential support need in workflows like Spherene) *and* are **downward-facing** (based on their normal vector).

This approach is useful because the exact method for defining supports can vary between slicers (a "slicer dialect"). The script uses a template file derived from your preferred slicer, ensuring better compatibility. While initially developed with [Spherene](https://www.spherene.ch) geometry workflows in mind, it can be used with any Rhino mesh that has appropriate vertex colors or where point-based support definition is desired.

## Features

* Takes a Rhino mesh with green vertex coloring for support locations and optionally selected Rhino points as input.
* Identifies faces needing support based on point proximity and/or vertex color/normal criteria.
* Allows configuration of color thresholds, normal direction, and point neighborhood expansion via variables in the script.
* Uses a customizable template `.3mf` structure for broad slicer compatibility.
* Outputs a ready-to-use `.3mf` file with painted supports and print presets (Customizable).

## Requirements

1. **Rhino 8:** CPython
2. **Python Packages:**
    * `numpy`
    * `scipy`
    (The script includes `# r: numpy` and `# r: scipy` directives to help Rhino 8 manage these.)
3. **Template Folder:** A correctly configured `template` folder (see Setup below).

## Tested Slicers
- Bambu Studio (Provided template file is from Bambu Studio with Bambu Lab A1 preset)
- Prusa Slicer
- Orca Slicer

## Setup

1. **Get Files:** Place the Python script file (e.g., `cpython_create_colored3mf.py`) and the `template` folder in the same directory on your computer.
2. **Configure Template for your printer (One-Time Setup):** This step ensures the output `.3mf` matches your slicer's preset. You only need to do this once per slicer/setting profile or you can use the exported .3mf as is and change your printer settings in your slicer everytime you import it.
    * **Export Base `.3mf`:** In your slicer (e.g., Bambu Studio, PrusaSlicer, Orca Slicer), configure your desired settings (printer, material, quality) and export a simple project as a `.3mf` file (e.g., `my_settings.3mf`).
    * **Rename to `.zip`:** Change the file extension to `.zip` (e.g., `my_settings.zip`).
    * **Extract Archive:** Extract the contents of the `.zip` file into a new folder.
    * **Rename Folder:** Rename this extracted folder to exactly `template`.
    * **Locate Model File:** Inside `template`, find the model file, typically `3D/Objects/object_1.model`.
    * **Edit Model File:** Open `object_1.model` in a text editor.
        * Replace **all** `<vertex .../>` lines and replace them with: `%vertices%`
        * Replace **all** `<triangle .../>` lines and replace them with: `%triangles%`
    * **Save:** Save the modified `object_1.model`.
    * **Verify:** Ensure this `template` folder (containing the modified structure) is in the same directory as the Python script. *(Tip: Compare with the provided example template).*

## Configuration (In Script)

Before running, you can adjust parameters directly within the Python script file. Open it in a text editor or the Rhino `ScriptEditor`. Find the `--- Configuration ---` section near the top:

* `VERBOSE_MODE = False`: Set to `True` for detailed console output during script execution (useful for debugging).
* `GREEN_THRESHOLD = 200`: Minimum green value (0-255) for a vertex to be considered 'green' for the color-based support criteria.
* `NORMAL_Z_THRESHOLD = -0.8`: Maximum Z-component of a face normal (-1.0 is straight down) for it to be considered 'downward-facing'.
* `NEIGHBORHOOD_EXPANSION_STEPS = 0`: How many layers of neighboring faces to add around the initial point-based support faces (0 = none, 1 = direct neighbors, etc.). More steps (1-2) helps when using tree supports.
* `SUPPORT_ATTRIBUTE_VALUE = "4"`: **Crucial for slicer compatibility.** This is the value assigned in the `.3mf` to mark faces for support. Bambu Studio uses the value values `"4"` which also works in Prusa Slicer and Orca Slicer. Other slicers might use `"all"`, `"true"`, `"painted"`. Check a `.3mf` saved using  *your slicer* with manually painted supports to see how painted supports are managed.
* `OUTPUT_BASENAME = "output_supports"`: The base name for the output file (saved as `output_supports.3mf`).
* `TEMP_DIR_NAME = "support_temp_processing"`: Name of the temporary folder used during processing.

## Usage

### Spherene Workflow

1. **Prepare Geometry:**
    * Compute a spherene result
        * In the compute dialog use the **RGB Color Scheme*** and activate **Support Points** (Generates greenish vertex coloring where supports might be needed and places support points).
2. **Select Spherene Result:**
    * The **Spherene Solid Mesh**.
    * *(Optional)* Add the support points to the selection.
3. **Run Script:**
    * Open the Rhino **ScriptEditor** (command: `ScriptEditor`).
    * Open the Python script file, cpython_create_colored3mf.py.
    * Run the script (e.g., click the "Play" button).
4. **Check Output:** The script will print progress. If successful, it creates a `.3mf` file (e.g., `output_supports.3mf`) in the same directory as the script.
5. **Open in Slicer:** Open the generated `.3mf` file as a project in your slicer. You should see the areas identified by the script highlighted or "painted" for support.
    - Make sure (In Bambu Studio...) **Normal(Manual)** or **Tree(Manual)** is activated
6. Slice plate
### Custom Workflow

1. **Prepare Geometry:**
    * Have your target **Mesh object** ready in Rhino.
    * If using color-based supports, ensure the mesh has **Vertex Colors** assigned (e.g., greenish colors where supports might be needed. 
    * Create **Point objects** in Rhino at the locations where you want supports anchored 
2.  **Select Objects:**:
    *   The **Mesh object** with/without vertex coloring.
    *   *(Optional)* Add **Points** you created for support locations to the selection.
3. **Run Script:**
    * Open the Rhino **ScriptEditor** (command: `ScriptEditor`).
    * Open the Python script file, cpython_create_colored3mf.py.
    * Run the script (e.g., click the "Play" button).
4. **Check Output:** The script will print progress. If successful, it creates a `.3mf` file (e.g., `output_supports.3mf`) in the same directory as the script.
5. **Open in Slicer:** Open the generated `.3mf` file as a project in your slicer. You should see the areas identified by the script highlighted or "painted" for support.
    - Make sure (In Bambu Studio...) **Normal(Manual)** or **Tree(Manual)** is activated
6. Slice plate

## Slicer Configuration (Important!)

* **Enable Supports:** Ensure supports are enabled in your slicer settings.
* **Support Type:** You **must** enable **Normal Manual** or **Tree Manual** support generation that specifically respects painted areas. For best results, set the support placement option to "On build plate only".

## Troubleshooting

* **"Template file/directory not found" Error:** Ensure the `template` folder exists, is named correctly, is in the same directory as the script, and the One-Time Setup was completed.
* **Supports Not Appearing/Incorrect in Slicer:**
    * Verify **Slicer Configuration** (Manual supports enabled?).
    * Double-check the **Template Setup** (correct replacements of `%vertices%`, `%triangles%`?).
    * Adjust the `SUPPORT_ATTRIBUTE_VALUE` in the script's Configuration section to match what your slicer expects.
* **Permission Errors:** Check file permissions for the script's directory if you get errors about creating/deleting the temporary folder or writing the output file.
* **Color/Normal Supports Not Working:** Ensure your mesh has vertex colors and that the `GREEN_THRESHOLD` and `NORMAL_Z_THRESHOLD` values are appropriate for your mesh's coloring and orientation. Use `VERBOSE_MODE = True` to see details in the console.

## License

This project is licensed under the **MIT License**.
