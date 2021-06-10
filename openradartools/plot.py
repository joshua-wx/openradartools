
import numpy as np
import matplotlib.colors as colors

def _adjust_ncarpid_scheme_colorbar_for_pyart(cb):
    """
    Generate colorbar for the hydrometeor classification.
    """
    cb.set_ticks(np.linspace(0.5, 19.5, 20))
    cb.ax.set_yticklabels(
        [
        "nodata","Cloud","Drizzle","Light_Rain","Moderate_Rain","Heavy_Rain",
           "Hail","Rain_Hail_Mixture","Graupel_Small_Hail","Graupel_Rain",
           "Dry_Snow", "Wet_Snow", "Ice_Crystals", "Irreg_Ice_Crystals",
           "Supercooled_Liquid_Droplets", "Flying_Insects", "Second_Trip", "Ground_Clutter",
           "misc1", "misc2"
        ]
    )
    cb.ax.set_ylabel("")
    cb.ax.tick_params(length=0)
    return cb

def _adjust_csuhca_scheme_colorbar_for_pyart(cb):
    """
    Generate colorbar for the hydrometeor classification.
    """
    cb.set_ticks(np.linspace(0.5, 10.5, 11))
    cb.ax.set_yticklabels(
        [
            "None",
            "Driz",
            "Rain",
            "IceCry",
            "IceAgg",
            "W Snow",
            "V Ice",
            "LD Gpl",
            "HD Gpl",
            "Hail",
            "Big Dp",
        ]
    )
    cb.ax.set_ylabel("")
    cb.ax.tick_params(length=0)
    return cb

def csu_color_map():

    color_list = [
        "White", #None
        "LightBlue", #Driz
        "SteelBlue", #Rain
        "MediumBlue", #IceCry
        "Plum", #IceAgg
        "MediumPurple", #W Snow
        "m", #V Ice
        "Green", #LD Gpl
        "YellowGreen", #HD Gpl
        "Gold", #Hail
        "Red", #Big Dp
        ]
    
    return colors.ListedColormap(color_list)

def ncar_color_map():

    color_list = [
        "White", #no data
        "lightgray", #cloud
        "LightBlue", #drizzle
        "SteelBlue", #light rain
        "RoyalBlue", #moderate rain
        "Navy", #heavy rain
        "fuchsia", #hail
        "darkviolet", #rain hail
        "hotpink", #Graupel_Small_Hail
        "crimson", #Graupel_Rain
        "lightgreen", #Dry_Snow 
        "limegreen", #Wet_Snow
        "green", #Ice_Crystals
        "seagreen", #Irreg_Ice_Crystals
        "Red", #Supercooled_Liquid_Droplets
        "orange", #Flying_Insects
        "chocolate", #Second_Trip
        "sienna", #Ground_Clutter
        "cyan", #misc1
        "teal", #misc2 
        ]
    
    return colors.ListedColormap(color_list)