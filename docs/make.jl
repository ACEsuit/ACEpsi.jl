using ACEpsi
using Documenter

DocMeta.setdocmeta!(ACEpsi, :DocTestSetup, :(using ACEpsi); recursive=true)

makedocs(;
    modules=[ACEpsi],
    authors="Christoph Ortner <christohortner@gmail.com> and contributors",
    repo="https://github.com/ACEsuit/ACEpsi.jl/blob/{commit}{path}#{line}",
    sitename="ACEpsi.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ACEsuit.github.io/ACEpsi.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ACEsuit/ACEpsi.jl",
    devbranch="main",
)
