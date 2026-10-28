-- Pandoc Lua filter for the PDF rendering of the E1-small filing snapshot.
-- Layout only (no text change): column widths of wide tables follow the
-- square root of each column's longest cell, and inline code may break
-- after _ . / = : - so long identifiers do not overflow the column.

local function cell_len(cell)
  return #pandoc.utils.stringify(cell.contents)
end

function Table(tbl)
  local ncol = #tbl.colspecs
  if ncol < 3 then return nil end
  local maxl = {}
  for i = 1, ncol do maxl[i] = 1 end
  local function scan(rows)
    for _, row in ipairs(rows) do
      for i, cell in ipairs(row.cells) do
        if i <= ncol then maxl[i] = math.max(maxl[i], cell_len(cell)) end
      end
    end
  end
  scan(tbl.head.rows)
  for _, body in ipairs(tbl.bodies) do scan(body.body) end
  local total = 0
  for i = 1, ncol do maxl[i] = math.sqrt(maxl[i]); total = total + maxl[i] end
  for i = 1, ncol do
    tbl.colspecs[i] = {tbl.colspecs[i][1], 0.98 * maxl[i] / total}
  end
  return tbl
end

local ESC = {["\\"] = "\\textbackslash{}", ["#"] = "\\#", ["$"] = "\\$", ["%"] = "\\%",
             ["&"] = "\\&", ["_"] = "\\_", ["{"] = "\\{", ["}"] = "\\}",
             ["~"] = "\\textasciitilde{}", ["^"] = "\\textasciicircum{}"}

local function tex_escape(ch)  -- one character, escaped exactly once
  return ESC[ch] or ch
end

function Code(el)
  if FORMAT ~= "latex" then return nil end
  local out = {}
  for ch in el.text:gmatch(utf8.charpattern) do
    local t = tex_escape(ch)
    if ch:match("[_%./=:%-]") then t = t .. "\\allowbreak{}" end
    out[#out + 1] = t
  end
  return pandoc.RawInline("latex", "\\texttt{" .. table.concat(out) .. "}")
end
