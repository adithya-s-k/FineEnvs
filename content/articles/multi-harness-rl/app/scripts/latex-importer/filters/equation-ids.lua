--[[
Pandoc Lua filter to add IDs to equations using KaTeX \htmlId syntax

This filter processes display math equations and inline math that contain 
\label{} commands, and wraps them with \htmlId{clean-id}{content} for KaTeX.

Requirements:
- KaTeX renderer with trust: true option
- Equations with \label{} commands in LaTeX
--]]

-- Function to clean identifier strings (remove prefixes and colons)
function clean_identifier(id_str)
    if id_str and type(id_str) == "string" then
        -- Remove common prefixes and replace colons with dashes
        local clean = id_str
            :gsub("^(eq|equation):", "")  -- Remove eq: prefix
            :gsub(":", "-")               -- Replace colons with dashes
            :gsub("[^a-zA-Z0-9_-]", "-")  -- Replace other problematic chars
            :gsub("-+", "-")              -- Collapse multiple dashes
            :gsub("^-", "")               -- Remove leading dash
            :gsub("-$", "")               -- Remove trailing dash
        
        -- Ensure we don't have empty identifiers
        if clean == "" then
            clean = id_str:gsub(":", "-")
        end
        
        return clean
    end
    return id_str
end

-- Process Math elements (both inline and display)
function Math(el)
    local math_content = el.text
    
    -- Look for \label{...} commands in the math content
    local label_match = math_content:match("\\label%{([^}]+)%}")
    
    if label_match then
        -- Clean the identifier
        local clean_id = clean_identifier(label_match)
        
        -- Remove the \label{} command from the math content
        local clean_math = math_content:gsub("\\label%{[^}]+%}", "")
        
        -- Clean up any extra whitespace or line breaks that might remain
        clean_math = clean_math:gsub("%s*$", ""):gsub("^%s*", "")
        
        -- Handle different equation environments appropriately
        -- For align environments, preserve them as they work with KaTeX
        local has_align = clean_math:match("\\begin%{align%}")
        
        if has_align then
            -- For align environments, we keep the structure and add ID as an attribute
            -- KaTeX supports align environments natively
            clean_math = clean_math:gsub("\\begin%{align%}", "\\begin{align}")
            clean_math = clean_math:gsub("\\end%{align%}", "\\end{align}")
        else
            -- Remove other equation environments that don't work well with \htmlId
            clean_math = clean_math:gsub("\\begin%{equation%}", ""):gsub("\\end%{equation%}", "")
            clean_math = clean_math:gsub("\\begin%{equation%*%}", ""):gsub("\\end%{equation%*%}", "")
            clean_math = clean_math:gsub("\\begin%{align%*%}", ""):gsub("\\end%{align%*%}", "")
        end
        
        -- Clean up any remaining whitespace
        clean_math = clean_math:gsub("%s*$", ""):gsub("^%s*", "")
        
        local new_math
        if has_align then
            -- For align environments, KaTeX doesn't support \htmlId with align
            -- Instead, we add a special marker that the post-processor will convert to a span
            -- This span will serve as an anchor for references
            new_math = "%%ALIGN_ANCHOR_ID{" .. clean_id .. "}%%\n" .. clean_math
        else
            -- For other math, wrap with \htmlId{}
            new_math = "\\htmlId{" .. clean_id .. "}{" .. clean_math .. "}"
        end
        
        -- Return new Math element with the updated content
        return pandoc.Math(el.mathtype, new_math)
    end
    
    -- Return unchanged if no label found
    return el
end

-- Optional: Process RawInline elements that might contain LaTeX math
function RawInline(el)
    if el.format == "latex" or el.format == "tex" then
        local content = el.text
        
        -- Look for equation environments with labels
        local label_match = content:match("\\label%{([^}]+)%}")
        
        if label_match then
            local clean_id = clean_identifier(label_match)
            
            -- For raw LaTeX, we might need different handling
            -- This is a simplified approach - adjust based on your needs
            local clean_content = content:gsub("\\label%{[^}]+%}", "")
            
            if clean_content:match("\\begin%{equation") or clean_content:match("\\begin%{align") then
                -- For equation environments, we might need to wrap differently
                -- This depends on how your KaTeX setup handles equation environments
                return pandoc.RawInline(el.format, clean_content)
            end
        end
    end
    
    return el
end

-- Optional: Process RawBlock elements for display equations
function RawBlock(el)
    if el.format == "latex" or el.format == "tex" then
        local content = el.text
        
        -- Look for equation environments with labels
        local label_match = content:match("\\label%{([^}]+)%}")
        
        if label_match then
            local clean_id = clean_identifier(label_match)
            local clean_content = content:gsub("\\label%{[^}]+%}", "")
            
            -- For block equations, we might want to preserve the structure
            -- but add the htmlId functionality
            return pandoc.RawBlock(el.format, clean_content)
        end
    end
    
    return el
end
