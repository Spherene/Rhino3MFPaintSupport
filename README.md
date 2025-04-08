# Rhino3MFPaintSupport

The purpose of this script is to generated support for Spherene geometry at the support points. Since this is highly printer dependent we deliever this functionality as a script.
See http:/spherene.ch howto generate spherene geometry


This is a CPython Script for Rhino8. The purpose of this script is to create support in a 3mf Files. It seems that each slicer has a bit of a dialect for marking the support in the 3mf. If you need to change than change the line 24 in the python script:
    fmt =  '\t<triangle v1="{}" v2="{}" v3="{}" paint_supports="4"/>\n' maybe the number "4" has to be replaced by the text "all".

Steps to success:
One-Time setup:
0. Save a simple project with your settings as a 3mf. Substitute the 3mf extension by zip and extract the archive. Rename the folder to template. Navigate to template/3D/Objects/object_1.model and substitute all <vertex .../> lines by the line %vertices%. Substitute all <triangle .../> lines by the line %triangles%. (If it is unclear have a look at the example template folder we provided)

1. Calculate a Spherene with Support Points. Use the RGB color scheme (greenish, faces where the normals is mostly parallel -z will have "paint_support")
2. Select the solid and the Support Points.
3. Start the script cpython_create_colored3mf.py from within ScriptEditor (openable by ScriptEditor Rhino Command)

4. Open the 3mf file as project in your slicer 


Remarks:
Do create the support you have to switch on manual normal support. (tree support seems not to be an option since it does not "see" all support markers (observerd in bambu studio))
