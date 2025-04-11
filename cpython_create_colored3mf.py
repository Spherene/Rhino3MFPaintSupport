#! python 3
# r: numpy
# r: scipy
# -----------------------------------------------------------------------------------------
# Name:         Rhino 3MF Support Painter
# Purpose:      Generates a .3mf file with painted supports based on selected points
#               and/or vertex color/face normal criteria in Rhino 8.
#
# Author:       spherene AG
#
# Created:      2025-04-07
# Modified:     2025-04-11
# Version:      1.0.0
#
# Requirements: Rhino 8, numpy, scipy
#
# License:      MIT License
#
#               Copyright (c) 2025 spherene AG
#
#               Permission is hereby granted, free of charge, to any person obtaining a copy
#               of this software and associated documentation files (the "Software"), to deal
#               in the Software without restriction, including without limitation the rights
#               to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#               copies of the Software, and to permit persons to whom the Software is
#               furnished to do so, subject to the following conditions:
#
#               The above copyright notice and this permission notice shall be included in all
#               copies or substantial portions of the Software.
#
#               THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#               IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#               FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#               AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#               LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#               OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#               SOFTWARE.
# -----------------------------------------------------------------------------------------

import numpy as np
import rhinoscriptsyntax as rs
import Rhino
import io
import os
import shutil
import sys
import pathlib
from scipy.spatial import KDTree

# --- Configuration ---
# Set VERBOSE_MODE to True for detailed developer/debug output in the console.
VERBOSE_MODE = False

# === Support Criteria ===
# Threshold for green vertex color detection (0-255). Higher values mean 'more green'.
GREEN_THRESHOLD = 200
# Max Z component for a face normal to be considered 'downward-facing' (-1.0 is straight down).
NORMAL_Z_THRESHOLD = -0.8
# Number of neighborhood expansion steps around point-based supports.
# 0 = Only faces directly near points. 1 = Add direct neighbors, 2 = Add neighbors of neighbors, etc.
NEIGHBORHOOD_EXPANSION_STEPS = 0 # Set to 0 to disable expansion. 1-2 helps with tree supports

# === Output Configuration ===
# The attribute value used in the .model file to mark supported faces.
# Common values: "4" (Bambu), "true", "painted". Check your slicer's .3mf format if needed.
SUPPORT_ATTRIBUTE_VALUE = "4"
# Base name for the output .3mf file (will be saved as output_basename.3mf)
OUTPUT_BASENAME = "output_supports"
# Name for the temporary directory used during processing.
TEMP_DIR_NAME = "support_temp_processing"
# --- End Configuration ---


# === Core Functions ===

def generate_xml(template_model_path, output_model_path, points, unsupported_faces, supported_faces):
    """Reads template, inserts vertex/face data, and writes the new model file."""
    print(f"Generating model file: {output_model_path}")
    try:
        with open(template_model_path, 'r', encoding='utf-8') as f:
            ins = f.read()
    except FileNotFoundError:
        print(f"Error: Template model file not found at {template_model_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading template model file {template_model_path}: {e}")
        sys.exit(1)

    # Prepare vertex data
    fmt_vertex = '\t<vertex x="{}" y="{}" z="{}"/>\n'
    vertex_stream = io.StringIO()
    for v in points:
        vertex_stream.write(fmt_vertex.format(*v))
    ins = ins.replace("%vertices%", vertex_stream.getvalue())

    # Prepare face data (unsupported first, then supported)
    fmt_unsupported = '\t<triangle v1="{}" v2="{}" v3="{}"/>\n'
    # Use the configured support attribute value
    fmt_supported = f'\t<triangle v1="{{}}" v2="{{}}" v3="{{}}" paint_supports="{SUPPORT_ATTRIBUTE_VALUE}"/>\n'
    face_stream = io.StringIO()
    for f in unsupported_faces:
        face_stream.write(fmt_unsupported.format(*f))
    for f in supported_faces:
        face_stream.write(fmt_supported.format(*f))
    ins = ins.replace("%triangles%", face_stream.getvalue())

    # Write the output file
    try:
        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
        with open(output_model_path, "w", encoding='utf-8', newline='\n') as f:
            f.write(ins)
        print("Model file generated successfully.")
    except Exception as e:
        print(f"Error writing model file {output_model_path}: {e}")
        try:
            # Attempt to find the temp directory by going up from the model file path
            temp_dir = os.path.dirname(os.path.dirname(os.path.dirname(output_model_path)))
            if os.path.exists(temp_dir) and os.path.basename(temp_dir) == TEMP_DIR_NAME: # Safety check
                 shutil.rmtree(temp_dir)
                 print(f"Cleaned up temporary directory {temp_dir} due to write error.")
        except Exception as cleanup_e:
             print(f"Also failed to cleanup temp directory during error handling: {cleanup_e}")
        sys.exit(1)


def get_neighbourhood(target_indices, all_faces):
    """Finds all faces that contain any of the target vertex indices."""
    if len(target_indices) == 0:
        return np.zeros(len(all_faces), dtype=bool)
    # Ensure target_indices is a NumPy array for isin
    target_indices_arr = np.asarray(target_indices)
    if target_indices_arr.size == 0:
        return np.zeros(len(all_faces), dtype=bool)
    is_face_member = np.isin(all_faces, target_indices_arr)
    is_neighbor_face = np.any(is_face_member, axis=1)
    return is_neighbor_face


def get_green_downward_faces(mesh_geo_original, face_definitions_original):
    """
    Identifies faces based on vertex color and normal using the ORIGINAL mesh state
    BEFORE vertex combination.
    Returns a boolean array aligned with face_definitions_original.
    """
    print(f"Identifying 'green' (G>{GREEN_THRESHOLD}) and downward-facing (Nz<{NORMAL_Z_THRESHOLD}) faces (using original mesh state)...")
    combined_condition = np.zeros(len(face_definitions_original), dtype=bool) # Default to no faces

    # 1. Face Normals (calculated on the original geometry state)
    downward_facing_cond = np.zeros(len(face_definitions_original), dtype=bool) # Default if calculation fails
    if not mesh_geo_original.FaceNormals.ComputeFaceNormals():
         print("Warning: Could not compute face normals on original mesh state.")
    else:
        normals = np.array([tuple(n) for n in mesh_geo_original.FaceNormals])
        # Ensure normals array matches face count
        if len(normals) == len(face_definitions_original):
            downward_facing_cond = normals[:, 2] < NORMAL_Z_THRESHOLD
            num_downward = np.sum(downward_facing_cond)
            print(f"Found {num_downward} faces potentially pointing downward.")
            if VERBOSE_MODE:
                 print(f">>> [Verbose] Downward condition (first 50): {downward_facing_cond[:50]}")
        else:
            print(f"Warning: Normal count ({len(normals)}) mismatch with face count ({len(face_definitions_original)}). Skipping downward check.")


    # 2. Vertex Colors (Green) - Accessed BEFORE CombineIdentical could affect them
    is_face_green = np.zeros(len(face_definitions_original), dtype=bool) # Default if calculation fails
    if mesh_geo_original.VertexColors.Count == 0:
         print("Warning: Mesh does not have any vertex colors defined (checked before CombineIdentical).")
    elif mesh_geo_original.VertexColors.Count != mesh_geo_original.Vertices.Count:
        print(f"Warning: Original vertex color count ({mesh_geo_original.VertexColors.Count}) does not match original vertex count ({mesh_geo_original.Vertices.Count}). Skipping green check.")
    else:
        # Proceed with green check using original vertex colors
        colors = mesh_geo_original.VertexColors
        greens = np.array([c.G for c in colors])
        is_vertex_green = greens > GREEN_THRESHOLD
        num_green_vertices = np.sum(is_vertex_green)
        print(f"Found {num_green_vertices} 'green' vertices (checked before CombineIdentical).")

        try:
            # Map original vertex green condition to original faces
            face_has_green_vertex = is_vertex_green[face_definitions_original]
            is_face_green = np.any(face_has_green_vertex, axis=1)
            num_green_faces = np.sum(is_face_green)
            print(f"Found {num_green_faces} faces with at least one 'green' vertex.")
            if VERBOSE_MODE:
                print(f">>> [Verbose] Vertex is green condition (first 50): {is_vertex_green[:50]}")
                print(f">>> [Verbose] Face is green condition (first 50): {is_face_green[:50]}")

        except IndexError as e:
            print(f"Error: Indexing error while mapping vertex colors to faces: {e}. Skipping green check.")
            # is_face_green remains the default array of False

    # 3. Combine conditions (Face must be BOTH green AND downward-facing)
    # Only combine if both checks produced valid boolean arrays of the correct length
    if len(is_face_green) == len(downward_facing_cond):
        combined_condition = is_face_green & downward_facing_cond
        num_final = np.sum(combined_condition)
        print(f"Found {num_final} faces that are both 'green' and downward-facing.")
        if VERBOSE_MODE:
             print(f">>> [Verbose] Combined green & downward condition (first 50): {combined_condition[:50]}")
    else:
        print("Warning: Could not combine green and downward conditions due to array length mismatch.")

    return combined_condition


def get_supported_faces_from_points(mesh_vertices_combined, all_faces_combined, support_points):
    """
    Identifies faces near points using the COMBINED mesh state.
    Returns a boolean array aligned with all_faces_combined.
    """
    if len(support_points) == 0:
        print("No support points provided. Skipping point-based support finding.")
        return np.zeros(len(all_faces_combined), dtype=bool)

    print("Finding faces closest to support points using KDTree (on combined mesh)...")
    try:
        tree = KDTree(mesh_vertices_combined) # Use combined vertices for KDTree
        distances, nearest_vertex_indices = tree.query(support_points)
    except Exception as e:
        print(f"Error during KDTree query: {e}")
        return np.zeros(len(all_faces_combined), dtype=bool)


    if VERBOSE_MODE:
        print(">>> [Verbose] KDTree Query Results:")
        print(f">>> Distances (support point to nearest combined mesh vertex): {distances}")
        print(f">>> Indices (of nearest combined mesh vertex): {nearest_vertex_indices}")

    # Use combined faces for neighborhood finding
    total_cond = get_neighbourhood(nearest_vertex_indices, all_faces_combined)
    initial_count = np.sum(total_cond)
    print(f"Found {initial_count} initial faces near support points.")

    # --- Neighborhood Expansion (uses combined faces) ---
    if NEIGHBORHOOD_EXPANSION_STEPS > 0:
        print(f"Performing {NEIGHBORHOOD_EXPANSION_STEPS} neighborhood expansion step(s)...")
        current_total_cond = total_cond.copy()
        expansion_step_count = 0
        for j in range(NEIGHBORHOOD_EXPANSION_STEPS):
            expansion_step_count = j + 1
            step_start_count = np.sum(current_total_cond)
            # Find unique vertices from currently selected faces
            indices_in_current_faces = np.unique(all_faces_combined[current_total_cond].flatten())
            # Find neighbor faces of these vertices
            new_neighbor_cond = get_neighbourhood(indices_in_current_faces, all_faces_combined)
            # Combine with existing condition
            updated_total_cond = np.logical_or(current_total_cond, new_neighbor_cond)
            step_end_count = np.sum(updated_total_cond)

            if VERBOSE_MODE: print(f">>> [Verbose] Expansion step {j+1}/{NEIGHBORHOOD_EXPANSION_STEPS}: In:{step_start_count} NewNeighbours:{np.sum(new_neighbor_cond)} Out:{step_end_count}")

            if step_end_count == step_start_count:
                 print(f"  No new faces added in step {j+1}. Stopping expansion early.")
                 break # Stop if expansion yields no new faces

            current_total_cond = updated_total_cond # Update the total for the next iteration

        total_cond = current_total_cond # Assign the final expanded condition
        print(f"Finished expansion. Total faces after {expansion_step_count} step(s): {np.sum(total_cond)}")
    # --- End Expansion ---

    num_supported = np.sum(total_cond)
    print(f"Found {num_supported} faces supported based on proximity to points (including expansion).")
    if VERBOSE_MODE: print(f">>> [Verbose] Final point-based support condition (first 50): {total_cond[:50]}")

    return total_cond


def normalize_line_endings(directory_path):
    """Ensures text-based files in the directory use Unix (LF) line endings."""
    if VERBOSE_MODE:
        print(f">>> [Verbose] Normalizing line endings (LF) in {directory_path} for .xml, .model, .rels, .txt files...")
    converted_count = 0
    try:
        for p in pathlib.Path(directory_path).rglob("*"):
            if p.is_file() and p.suffix.lower() in [".xml", ".model", ".txt", ".rels"]:
                try:
                    text = p.read_text(encoding="utf-8")
                    # Check for CR or CRLF, avoiding potential double replacement issues
                    if '\r\n' in text or ('\r' in text and '\n' not in text.replace('\r\n', '')):
                         new_text = text.replace('\r\n', '\n').replace('\r', '\n')
                         p.write_text(new_text, encoding="utf-8", newline='\n')
                         converted_count += 1
                except Exception as e:
                     print(f"Warning: Could not process file {p} for line ending normalization: {e}")
        if VERBOSE_MODE:
            print(f">>> [Verbose] Line ending normalization complete. {converted_count} files modified.")
    except Exception as e:
         print(f"Warning: Error during line ending normalization process: {e}")


# --- Main Script Execution ---
print("--- Starting Support Painting Script ---")
if VERBOSE_MODE:
    print(">>> Running in VERBOSE mode <<<")
    print(f">>> Config: GREEN_T={GREEN_THRESHOLD}, NORMAL_Z_T={NORMAL_Z_THRESHOLD}, EXP_STEPS={NEIGHBORHOOD_EXPANSION_STEPS}, SUPPORT_VAL='{SUPPORT_ATTRIBUTE_VALUE}'")

# --- Object Selection ---
rids = rs.SelectedObjects()
if not rids:
    print("Error: No objects selected. Please select exactly one mesh and optionally some points.")
    sys.exit(1)

support_points_coords = []
mesh_objects = []
print("Processing selected objects...")
for rid in rids:
    if rs.IsPoint(rid):
        support_points_coords.append(tuple(rs.PointCoordinates(rid)))
    elif rs.IsMesh(rid):
        mesh_objects.append(rid)

support_points_coords = np.array(support_points_coords)
print(f"Found {len(support_points_coords)} support points.")
if VERBOSE_MODE and len(support_points_coords) > 0:
    print(">>> [Verbose] Support Points Raw Data (Original Coords):")
    print(support_points_coords)

# --- Input Validation ---
if len(mesh_objects) != 1:
    print(f"\nError: Incorrect selection. Expected 1 mesh, found {len(mesh_objects)}.")
    sys.exit(1)

mesh_obj_id = mesh_objects[0] # Keep the ID for potential re-coercion if needed

# --- Mesh Processing ---
print("Processing mesh geometry...")
mesh_geo = rs.coercerhinoobject(mesh_obj_id).Geometry
if not mesh_geo:
     print(f"Error: Could not get mesh geometry from selected object ID: {mesh_obj_id}.")
     sys.exit(1)

# **Step 1: Get ORIGINAL state data needed for color/normal check**
print("Reading original mesh state for color/normal checks...")
# It might be safer to make a copy if CombineIdentical modifies in place significantly
# mesh_geo_original_copy = mesh_geo.DuplicateMesh() # Consider if CombineIdentical is destructive
mesh_geo_original_copy = mesh_geo.Duplicate() # Use Duplicate for general geometry
if not mesh_geo_original_copy or not isinstance(mesh_geo_original_copy, Rhino.Geometry.Mesh):
     print("Error: Could not duplicate original mesh geometry for checks.")
     sys.exit(1)

original_vertices_count = mesh_geo_original_copy.Vertices.Count
original_mesh_faces = np.array([[f.A, f.B, f.C] for f in mesh_geo_original_copy.Faces])
print(f"Original mesh state: {original_vertices_count} vertices, {len(original_mesh_faces)} faces.")

# **Step 2: Perform color/normal check BEFORE combining vertices**
cond_green_downward = get_green_downward_faces(mesh_geo_original_copy, original_mesh_faces) # Pass the copy

# **Step 3: Combine Identical Vertices on the main geometry object**
# Note: This might modify the 'mesh_geo' object in place.
print("Combining identical vertices...")
# Re-coerce just in case, though likely not necessary if mesh_geo is already the geometry
mesh_geo_for_combine = rs.coercerhinoobject(mesh_obj_id).Geometry
if not mesh_geo_for_combine:
    print("Error: Could not re-access mesh geometry before combining.")
    sys.exit(1)

original_vert_count_before_combine = mesh_geo_for_combine.Vertices.Count
if VERBOSE_MODE: print(f">>> [Verbose] Vertex count before CombineIdentical: {original_vert_count_before_combine}")
combined = mesh_geo_for_combine.Vertices.CombineIdentical(True, True)
combined_vert_count_after_combine = mesh_geo_for_combine.Vertices.Count
if VERBOSE_MODE: print(f">>> [Verbose] CombineIdentical result: {combined}. Vertex count after: {combined_vert_count_after_combine}")

# **Step 4: Get COMBINED state data for point checks and final output**
# Use the potentially modified mesh_geo_for_combine object
print("Reading combined mesh state for point checks and final output...")
combined_mesh_vertices = np.array([tuple(v) for v in mesh_geo_for_combine.Vertices])
combined_mesh_faces = np.array([[f.A, f.B, f.C] for f in mesh_geo_for_combine.Faces])
print(f"Combined mesh state: {len(combined_mesh_vertices)} vertices, {len(combined_mesh_faces)} faces.")

# **Important Sanity Check:** Verify face count hasn't changed unexpectedly
if len(original_mesh_faces) != len(combined_mesh_faces):
    print(f"Warning: Face count changed after CombineIdentical! Original={len(original_mesh_faces)}, Combined={len(combined_mesh_faces)}. Support combination might be inaccurate.")
    # Consider exiting if face count changes, as direct combination of boolean arrays becomes invalid
    # sys.exit(1) # Uncomment to make this a fatal error

# **Step 5: Normalize Z height (using COMBINED vertices)**
print("Normalizing Z height...")
# Check if there are vertices before attempting min()
if len(combined_mesh_vertices) == 0:
     print("Warning: Combined mesh has no vertices. Skipping normalization.")
     min_z = 0
     shift_z = np.array((0, 0, 0))
else:
    min_z = combined_mesh_vertices[:, 2].min()
    shift_z = np.array((0, 0, -min_z))
    combined_mesh_vertices += shift_z
    if len(support_points_coords) > 0: support_points_coords += shift_z # Shift points too
print(f"Normalized combined mesh and points Z height (minimum Z is now 0, shift applied: {shift_z[2]:.4f}).")
if VERBOSE_MODE:
    if len(combined_mesh_vertices)>0: print(f">>> [Verbose] Combined Vertices (Normalized, first 10): {combined_mesh_vertices[:10]}")
    if len(support_points_coords) > 0: print(f">>> [Verbose] Support Points (Normalized, first 10): {support_points_coords[:10]}")


# --- Identify Supported Faces ---
# **Step 6: Point-based check using COMBINED geometry**
# (cond_green_downward was already calculated in Step 2 using original geometry)
cond_points = get_supported_faces_from_points(combined_mesh_vertices, combined_mesh_faces, support_points_coords)

# **Step 7: Combine Conditions**
# Assumes face order/count consistency between original_mesh_faces and combined_mesh_faces
print("Combining support conditions (Green/Downward OR Near Points)...")
# Ensure boolean arrays are compatible for combination
if len(cond_green_downward) != len(cond_points):
     print("Error: Mismatch in face counts between color/normal check and point check results. Cannot combine conditions.")
     # Set final condition to only point-based if color check failed alignment
     print("Using only point-based results due to mismatch.")
     final_support_cond = cond_points
     # Alternatively, exit: sys.exit(1)
else:
     final_support_cond = cond_green_downward | cond_points

num_final_supported = np.sum(final_support_cond)

# **Step 8: Prepare final face lists using COMBINED face definitions**
# Check if final_support_cond is valid before indexing
if len(final_support_cond) == len(combined_mesh_faces):
    unsupported_faces_final = combined_mesh_faces[~final_support_cond]
    supported_faces_final = combined_mesh_faces[final_support_cond]
else:
    print("Error: Final support condition array size mismatch. Cannot determine final faces.")
    # Handle error: maybe output all as unsupported?
    unsupported_faces_final = combined_mesh_faces
    supported_faces_final = np.array([], dtype=combined_mesh_faces.dtype) # Empty array


# --- Support Summary ---
print(f"\n--- Support Summary ---")
print(f"Total Faces (Combined Mesh):  {len(combined_mesh_faces)}")
print(f"Green & Downward Faces:     {np.sum(cond_green_downward)}") # Based on original check
print(f"Point-Based Faces (w/ exp): {np.sum(cond_points)}")     # Based on combined check
print(f"Total Supported (Combined): {num_final_supported}")    # Sum of final boolean array
print(f"Total Unsupported:          {len(unsupported_faces_final)}") # Length of final list
if VERBOSE_MODE and len(supported_faces_final) > 0:
    print(f">>> [Verbose] Final Supported Face Defs (first 10): {supported_faces_final[:10]}")
if VERBOSE_MODE and len(unsupported_faces_final) > 0:
    print(f">>> [Verbose] Final Unsupported Face Defs (first 10): {unsupported_faces_final[:10]}")


# --- File Paths and Preparation ---
try:
     # Get script path safely
     cpath = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
except NameError:
     cpath = os.getcwd() # Fallback

template_dir = os.path.join(cpath, "template")
temp_out_dir = os.path.join(cpath, TEMP_DIR_NAME)

print(f"\nPreparing temporary directory: {temp_out_dir}")
try:
    if os.path.exists(temp_out_dir):
        shutil.rmtree(temp_out_dir)
        print("Removed existing temporary directory.")
except OSError as e:
    print(f"Warning: Could not remove existing temporary directory '{temp_out_dir}': {e}. Attempting to continue...")
except Exception as e:
    print(f"An unexpected error occurred during initial cleanup of {temp_out_dir}: {e}")


# --- Copy Template ---
try:
    # Ensure template directory exists before copying
    if not os.path.isdir(template_dir):
         print(f"Error: Template directory not found at '{template_dir}'. Please ensure it exists next to the script.")
         sys.exit(1)
    shutil.copytree(template_dir, temp_out_dir)
    print(f"Copied template files from '{template_dir}' to '{temp_out_dir}'.")
except Exception as e:
    print(f"Error copying template directory: {e}")
    sys.exit(1)


# --- Generate Model File (Using COMBINED vertices and FINAL face lists) ---
model_rel_path = os.path.join("3D", "Objects", "object_1.model")
template_model_path = os.path.join(template_dir, model_rel_path) # Path to template model file
output_model_path = os.path.join(temp_out_dir, model_rel_path) # Path to output model file in temp dir

# Check if template model file exists before generating
if not os.path.isfile(template_model_path):
    print(f"Error: Template model file not found inside template directory: {template_model_path}")
    sys.exit(1)

# Pass the COMBINED vertices and the FINAL derived supported/unsupported faces
generate_xml(template_model_path, output_model_path, combined_mesh_vertices, unsupported_faces_final, supported_faces_final)

# --- Normalize Line Endings ---
normalize_line_endings(temp_out_dir)


# --- Create Archive ---
archive_base_name = os.path.join(cpath, OUTPUT_BASENAME)
final_3mf_name = archive_base_name + ".3mf"

print(f"\nCreating 3MF archive: {final_3mf_name}")
try:
    if os.path.exists(final_3mf_name):
        os.remove(final_3mf_name)
        print(f"Removed existing output file: {final_3mf_name}")

    created_zip_path = shutil.make_archive(
        base_name=archive_base_name,
        format='zip',
        root_dir=temp_out_dir,
        base_dir="."
    )

    # Check if zip was created before renaming
    if not os.path.exists(created_zip_path):
         raise IOError(f"Failed to create zip archive at {created_zip_path}")

    os.rename(created_zip_path, final_3mf_name)
    print(f"Successfully created archive and renamed to: {final_3mf_name}")

except Exception as e:
    print(f"Error creating or renaming archive: {e}")
    # Attempt cleanup even if archiving fails
    try:
        if os.path.exists(temp_out_dir):
            print(f"Cleaning up temporary directory due to archive error: {temp_out_dir}")
            shutil.rmtree(temp_out_dir)
    except OSError as cleanup_e:
        print(f"Warning: Could not remove temporary directory '{temp_out_dir}' after archive error: {cleanup_e}")
    sys.exit(1)


# --- Final Cleanup ---
print(f"Cleaning up temporary directory: {temp_out_dir}")
try:
    if os.path.exists(temp_out_dir):
        shutil.rmtree(temp_out_dir)
        print("Temporary directory cleaned up successfully.")
    else:
        if VERBOSE_MODE: print(">>> [Verbose] Temporary directory was already removed or not created.")
except OSError as e:
    print(f"\nWarning: Could not automatically remove the temporary directory '{temp_out_dir}'.")
    print(f"Reason: {e}")
    print("You may need to delete it manually.")
except Exception as e:
    print(f"An unexpected error occurred during final cleanup: {e}")


print("\n--- Script Finished ---")
print(f"Output file created at: {final_3mf_name}")
