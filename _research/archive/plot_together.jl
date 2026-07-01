#=============================================================
SCRIPT TO PLOT FIGURES TOGETHER
=============================================================#

# Initialise project
using Pkg
# Activate the project
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
cd(@__DIR__)
using DrWatson
@quickactivate("UDE_FUNCTIONAL_FORMS")

# Name root
root = datadir("sims", "ude")
# Initialise empty array for png file names
png_files = String[]

# Read all files/folders in the root directory
for filename in readdir(root)
    # Only include directories
    if isdir(joinpath(root, filename))
        # Store path to plot if it exists
        plot_path = datadir("sims", "ude", filename, "prediction_plot.png")
        isfile(plot_path) && push!(png_files, plot_path)
    end
end

# Find number of Plots
n_plots = length(png_files)
# Plot three columns of Plots
n_cols = 3
n_rows = ceil(Int, n_plots/n_cols)

# Create subplot
subplots = [
    plot(load(pl), axis = nothing, ticks = nothing, border=:none, legend = false)
    for pl in png_files 
]

combined_plot = plot(subplots..., layout=(n_rows, n_cols), size=(2000, 400*n_rows), plot_title = "Hyperparameter comparison", dpi = 300)

# Save the combined plot
savefig(combined_plot, joinpath("figures", "HP_combined_plots.png"))