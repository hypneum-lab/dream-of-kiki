-- Pandoc Lua filter (layout only, no text change): render ~~struck~~
-- text with ulem's \sout instead of soul's \st, which drops characters
-- that are not in its fallback font (for example "τ") under xelatex.
function Strikeout(el)
  if FORMAT ~= "latex" then return nil end
  local out = {pandoc.RawInline("latex", "\\sout{")}
  for _, x in ipairs(el.content) do out[#out + 1] = x end
  out[#out + 1] = pandoc.RawInline("latex", "}")
  return out
end
