import Pkg; Pkg.add("Documenter"); Pkg.add("NNHelferlein")
using Documenter, NNHelferlein

makedocs(modules = [NNHelferlein],
         clean = false,
         assets = ["assets/favicon.ico"],
         sitename = "NNHelferlein.jl",
         authors = "Andreas Dominik",
         pages = [
                  "Introduction" => "index.md",
                  "Overview" => "overview.md",
                  "Examples" => "examples.md",
                  "API Reference" => "api.md",
                  "License" => "license.md"
                  ],
         #          # Use clean URLs, unless built as a "local" build
          html_prettyurls = false, #!("local" in ARGS),
          html_canonical = "https://andreasdominik.github.io/NNHelferlein.jl/stable/"
         )

deploydocs(
    # root   = "<current-directory>",
    target = "build",
    repo   = "github.com/andreasdominik/NNHelferlein.jl.git",
    branch = "gh-pages",
    # deps   = nothing | <Function>,
    # make   = nothing | <Function>,
    devbranch = "main",
    devurl = "dev",
    # versions = ["stable" => "v^", "v#.#", devurl => "dev"],
    push_preview    = false,
    # repo_previews   = repo,
    # branch_previews = branch
)
