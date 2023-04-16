import subprocess 
import os

def get_results():
    # Get the current working directory
    current_dir = os.getcwd()

    # Execute the Python script
    subprocess.call(["python3", "mains/main_affine_model.py"])

def compile_and_open():
    # Replace the following variables with your own file paths
    tex_file = "main.tex"
    pdf_viewer = "open"  # This is the command to open a file on macOS

    # Get the current working directory
    current_dir = os.getcwd()

    # Change the working directory to the 'tex' subdirectory
    os.chdir("tex")

    # Compile the .tex file using pdflatex
    subprocess.call(["pdflatex", "-interaction=nonstopmode", tex_file])

    # Open the resulting .pdf file using the default PDF viewer
    subprocess.call([pdf_viewer, tex_file.replace(".tex", ".pdf")])

    # Change the working directory back to the original directory
    os.chdir(current_dir)

def main():
    get_results()
    #compile_and_open()

if __name__ == "__main__":
    main()
