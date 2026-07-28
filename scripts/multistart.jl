#========================================================
SCRIPT TO DO MULTISTART AS A POST-PROCESSING STEP
=========================================================#  
using Pkg
# Activate the project
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
cd(@__DIR__)
using DrWatson
@quickactivate("UDE_FUNCTIONAL_FORMS")


# Call module
using UDE_FUNCTIONAL_FORMS

