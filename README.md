# som-patterns
A unified repo for both perceptual and phonological patterns.

This repo depends on the [semantic-features](https://github.com/oliviaguest/semantic-features) and  [phonological-features](https://github.com/oliviaguest/phonological-features) repos. They are [submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules). 
If you want to run any of the scripts here to regenerate stuff for yourself you will need to also grab those repos. Currently they are private. 

# standardise.py
It writes to the two subdirectories, which is a bit naughty, but at the moment can't be helped.

It takes the two pattern sets and makes sure they overlap 100% in terms of what concepts they represent. *And* in the case of semantic patterns, because they end up having some items removed, the script goes through and discoveres which features are never used and then deletes them. So features that are never on, are deleted, thus making the patterns shorter.
