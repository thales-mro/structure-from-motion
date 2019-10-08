from world_reconstruction import WorldReconstruction


def main():
    """
    Entrypoint for the code of project 03 Group 08 MO446/2sem2019
    """
    
    # Create the World Reconstruction object
    wr = WorldReconstruction()

    # Generate the 3D shape 
    wr.execute("input/i-2.mp4", "output/out.txt", operation=1, max_frames=-1, print_frames=True)


main()
