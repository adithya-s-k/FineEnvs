-- The article schema calls these authors/published; Pandoc calls them author/date.
-- Keep Pandoc's parsed metadata values so its writer escapes LaTeX characters.
function Meta(meta)
  if not meta.author and meta.authors then
    if pandoc.utils.type(meta.authors) == "List" then
      local names = pandoc.List()
      for _, author in ipairs(meta.authors) do
        names:insert(author.name or author)
      end
      meta.author = pandoc.MetaList(names)
    else
      meta.author = meta.authors
    end
  end
  if not meta.date and meta.published then meta.date = meta.published end
  -- Article-only author links and social handles are not publication metadata.
  -- Do not send them through citeproc as accidental @citation identifiers.
  local publication = {}
  for _, key in ipairs({"title", "subtitle", "author", "date", "abstract", "keywords",
      "lang", "bibliography", "references", "csl", "nocite", "link-citations"}) do
    publication[key] = meta[key]
  end
  return publication
end
