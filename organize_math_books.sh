#!/bin/bash

# Script to organize math book images into a consistent volume-based structure
# Author: OCR Compare Assistant
# Purpose: Reorganize images of math books (covers, intros, ToCs) into structured folders

set -e  # Exit on error

# Base directories
SOURCE_DIR="math-books"
TARGET_DIR="volumes"

# Create target directory
mkdir -p "$TARGET_DIR"

# Hebrew to English mapping for publishers
declare -A PUBLISHER_MAP=(
    ["בני גורן"]="goren"
    ["ארכימדס"]="archimedes"
    ["יואל גבע"]="geva"
)

# Hebrew to English mapping for volumes
declare -A VOLUME_MAP=(
    ["א"]="a"
    ["ב"]="b"
    ["ג"]="c"
    ["ד"]="d"
    ["ה"]="e"
)

# Hebrew to English mapping for grades
declare -A GRADE_MAP=(
    ["י"]="10"
    ["יא"]="11"
    ["יב"]="12"
)

# Function to clean filename and remove special characters
clean_filename() {
    local filename="$1"
    # Remove leading special characters and spaces
    echo "$filename" | sed 's/^[[:space:]‏]*//g' | sed 's/^[0-9]*//g'
}

# Function to determine category from filename
get_category() {
    local filename="$1"
    local clean_name=$(clean_filename "$filename")
    
    if [[ "$clean_name" =~ כריכה|cover|כותרת ]]; then
        echo "cover"
    elif [[ "$clean_name" =~ מבוא|פרומפט|intro ]]; then
        echo "intro"
    elif [[ "$clean_name" =~ תוכן|תוכן_עניינים|toc ]]; then
        echo "toc"
    else
        echo "unknown"
    fi
}

# Function to extract number from filename
get_number() {
    local filename="$1"
    local category="$2"
    
    # Extract number after underscore or space
    if [[ "$filename" =~ _([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    elif [[ "$filename" =~ \ ([0-9]+)\. ]]; then
        echo "${BASH_REMATCH[1]}"
    elif [[ "$filename" =~ $category.*([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo "1"
    fi
}

# Function to process image file
process_image() {
    local source_path="$1"
    local target_folder="$2"
    local filename=$(basename "$source_path")
    local extension="${filename##*.}"
    
    # Get category
    local category=$(get_category "$filename")
    
    if [[ "$category" == "unknown" ]]; then
        echo "  Warning: Unknown category for $filename, skipping..."
        return
    fi
    
    # Get number
    local number=$(get_number "$filename" "$category")
    
    # Create target filename
    local target_filename="${category}-${number}.${extension}"
    local target_path="$target_folder/$target_filename"
    
    # Copy file (avoid overwriting)
    if [[ -f "$target_path" ]]; then
        local counter=2
        while [[ -f "${target_folder}/${category}-${number}_${counter}.${extension}" ]]; do
            ((counter++))
        done
        target_path="${target_folder}/${category}-${number}_${counter}.${extension}"
    fi
    
    echo "  Copying: $filename -> $(basename "$target_path")"
    cp "$source_path" "$target_path"
}

# Process בני גורן (Beni Goren) books
echo "Processing בני גורן books..."

# Grade 10, Level 3
if [[ -d "$SOURCE_DIR/בני גורן/בני גורן תוכן עניינים/כיתה י 3 יחל" ]]; then
    echo "  Processing 3-10-goren volumes..."
    
    # Volume א (a)
    mkdir -p "$TARGET_DIR/3-10-goren-a"
    for img in "$SOURCE_DIR/בני גורן/בני גורן תוכן עניינים/כיתה י 3 יחל"/*כרך\ א*; do
        [[ -f "$img" ]] && process_image "$img" "$TARGET_DIR/3-10-goren-a"
    done
    
    # Volume ב (b)
    mkdir -p "$TARGET_DIR/3-10-goren-b"
    for img in "$SOURCE_DIR/בני גורן/בני גורן תוכן עניינים/כיתה י 3 יחל"/*כרך*ב*; do
        [[ -f "$img" ]] && process_image "$img" "$TARGET_DIR/3-10-goren-b"
    done
    
    # Volume ג (c)
    mkdir -p "$TARGET_DIR/3-10-goren-c"
    for img in "$SOURCE_DIR/בני גורן/בני גורן תוכן עניינים/כיתה י 3 יחל"/*כרך\ ג*; do
        [[ -f "$img" ]] && process_image "$img" "$TARGET_DIR/3-10-goren-c"
    done
fi

# Grade 10, Level 4
if [[ -d "$SOURCE_DIR/בני גורן/בני גורן תוכן עניינים/כיתה י 4 יחל" ]]; then
    echo "  Processing 4-10-goren volumes..."
    
    # Process each sub-volume folder
    for vol_dir in "$SOURCE_DIR/בני גורן/בני גורן תוכן עניינים/כיתה י 4 יחל"/בני\ ג.*; do
        if [[ -d "$vol_dir" ]]; then
            vol_name=$(basename "$vol_dir")
            if [[ "$vol_name" =~ _א$ ]]; then
                vol_letter="a"
            elif [[ "$vol_name" =~ _ב$ ]]; then
                vol_letter="b"
            elif [[ "$vol_name" =~ _ג$ ]]; then
                vol_letter="c"
            else
                continue
            fi
            
            mkdir -p "$TARGET_DIR/4-10-goren-$vol_letter"
            for img in "$vol_dir"/*.{JPG,jpg,jpeg,png,PNG}; do
                [[ -f "$img" ]] && process_image "$img" "$TARGET_DIR/4-10-goren-$vol_letter"
            done
        fi
    done
fi

# Grade 10, Level 5 
if [[ -d "$SOURCE_DIR/בני גורן/בני גורן תוכן עניינים/כיתה י 5 יחל" ]]; then
    echo "  Processing 5-10-goren volume..."
    mkdir -p "$TARGET_DIR/5-10-goren-a"
    for img in "$SOURCE_DIR/בני גורן/בני גורן תוכן עניינים/כיתה י 5 יחל"/*.{JPG,jpg,jpeg,png,PNG}; do
        [[ -f "$img" ]] && process_image "$img" "$TARGET_DIR/5-10-goren-a"
    done
fi

# Grade 11, Level 3
if [[ -d "$SOURCE_DIR/בני גורן/בני גורן תוכן עניינים/כיתה יא 3 יחל" ]]; then
    echo "  Processing 3-11-goren volumes..."
    
    # Volume א
    mkdir -p "$TARGET_DIR/3-11-goren-a"
    for img in "$SOURCE_DIR/בני גורן/בני גורן תוכן עניינים/כיתה יא 3 יחל"/*כרך\ א*; do
        [[ -f "$img" ]] && process_image "$img" "$TARGET_DIR/3-11-goren-a"
    done
    
    # Volume ב
    mkdir -p "$TARGET_DIR/3-11-goren-b"
    for img in "$SOURCE_DIR/בני גורן/בני גורן תוכן עניינים/כיתה יא 3 יחל"/*כרך*ב*; do
        [[ -f "$img" ]] && process_image "$img" "$TARGET_DIR/3-11-goren-b"
    done
    
    # Volume ג
    mkdir -p "$TARGET_DIR/3-11-goren-c"
    for img in "$SOURCE_DIR/בני גורן/בני גורן תוכן עניינים/כיתה יא 3 יחל"/*כרך*ג*; do
        [[ -f "$img" ]] && process_image "$img" "$TARGET_DIR/3-11-goren-c"
    done
fi

# Grade 11, Level 4
if [[ -d "$SOURCE_DIR/בני גורן/בני גורן תוכן עניינים/כיתה יא 4 יחל" ]]; then
    echo "  Processing 4-11-goren volumes..."
    
    for vol_dir in "$SOURCE_DIR/בני גורן/בני גורן תוכן עניינים/כיתה יא 4 יחל"/בני\ ג.*; do
        if [[ -d "$vol_dir" ]]; then
            vol_name=$(basename "$vol_dir")
            if [[ "$vol_name" =~ _א$ ]]; then
                vol_letter="a"
            elif [[ "$vol_name" =~ _ב$ ]]; then
                vol_letter="b"
            elif [[ "$vol_name" =~ _ג$ ]]; then
                vol_letter="c"
            else
                continue
            fi
            
            mkdir -p "$TARGET_DIR/4-11-goren-$vol_letter"
            for img in "$vol_dir"/*.{JPG,jpg,jpeg,png,PNG}; do
                [[ -f "$img" ]] && process_image "$img" "$TARGET_DIR/4-11-goren-$vol_letter"
            done
        fi
    done
fi

# Grade 11, Level 5
if [[ -d "$SOURCE_DIR/בני גורן/בני גורן תוכן עניינים/כיתה יא 5 יחל" ]]; then
    echo "  Processing 5-11-goren volumes..."
    
    # Volume ב_1
    mkdir -p "$TARGET_DIR/5-11-goren-b1"
    for img in "$SOURCE_DIR/בני גורן/בני גורן תוכן עניינים/כיתה יא 5 יחל"/*כרך\ ב_1*; do
        [[ -f "$img" ]] && process_image "$img" "$TARGET_DIR/5-11-goren-b1"
    done
    
    # Volume ב_2
    mkdir -p "$TARGET_DIR/5-11-goren-b2"
    for img in "$SOURCE_DIR/בני גורן/בני גורן תוכן עניינים/כיתה יא 5 יחל"/*כרך\ ב_2*; do
        [[ -f "$img" ]] && process_image "$img" "$TARGET_DIR/5-11-goren-b2"
    done
fi

# Grade 12, Level 3
if [[ -d "$SOURCE_DIR/בני גורן/בני גורן תוכן עניינים/כיתה יב 3 יחל" ]]; then
    echo "  Processing 3-12-goren volumes..."
    mkdir -p "$TARGET_DIR/3-12-goren-geometry"
    for img in "$SOURCE_DIR/בני גורן/בני גורן תוכן עניינים/כיתה יב 3 יחל"/*.{JPG,jpg,jpeg,png,PNG}; do
        [[ -f "$img" ]] && process_image "$img" "$TARGET_DIR/3-12-goren-geometry"
    done
fi

# Grade 12, Level 5
if [[ -d "$SOURCE_DIR/בני גורן/בני גורן תוכן עניינים/כיתה יב 5 יחל" ]]; then
    echo "  Processing 5-12-goren volumes..."
    
    # Volume ג_1
    if [[ -d "$SOURCE_DIR/בני גורן/בני גורן תוכן עניינים/כיתה יב 5 יחל/ג_1" ]]; then
        mkdir -p "$TARGET_DIR/5-12-goren-c1"
        for img in "$SOURCE_DIR/בני גורן/בני גורן תוכן עניינים/כיתה יב 5 יחל/ג_1"/*.{JPG,jpg,jpeg,png,PNG}; do
            [[ -f "$img" ]] && process_image "$img" "$TARGET_DIR/5-12-goren-c1"
        done
    fi
    
    # Volume ג_2
    if [[ -d "$SOURCE_DIR/בני גורן/בני גורן תוכן עניינים/כיתה יב 5 יחל/ג_2" ]]; then
        mkdir -p "$TARGET_DIR/5-12-goren-c2"
        for img in "$SOURCE_DIR/בני גורן/בני גורן תוכן עניינים/כיתה יב 5 יחל/ג_2"/*.{JPG,jpg,jpeg,png,PNG}; do
            [[ -f "$img" ]] && process_image "$img" "$TARGET_DIR/5-12-goren-c2"
        done
    fi
fi

# Process ארכימדס (Archimedes) books
echo -e "\nProcessing ארכימדס books..."

# Grade 10, Level 4 (471)
if [[ -d "$SOURCE_DIR/ארכימדס/ארכימדס/י 471" ]]; then
    echo "  Processing 4-10-archimedes volume..."
    mkdir -p "$TARGET_DIR/4-10-archimedes-a"
    for img in "$SOURCE_DIR/ארכימדס/ארכימדס/י 471"/*.{pdf,PDF}; do
        if [[ -f "$img" ]]; then
            filename=$(basename "$img")
            if [[ "$filename" =~ כריכה ]]; then
                cp "$img" "$TARGET_DIR/4-10-archimedes-a/cover-1.pdf"
            elif [[ "$filename" =~ מבוא ]]; then
                cp "$img" "$TARGET_DIR/4-10-archimedes-a/intro-1.pdf"
            elif [[ "$filename" =~ תוכן ]]; then
                if [[ "$filename" =~ ([0-9]+) ]]; then
                    num="${BASH_REMATCH[1]}"
                    cp "$img" "$TARGET_DIR/4-10-archimedes-a/toc-$num.pdf"
                fi
            fi
        fi
    done
fi

# Grade 10, Level 5 (571)
if [[ -d "$SOURCE_DIR/ארכימדס/ארכימדס/י 571" ]]; then
    echo "  Processing 5-10-archimedes volume..."
    mkdir -p "$TARGET_DIR/5-10-archimedes-a"
    
    # Process PDFs
    for img in "$SOURCE_DIR/ארכימדס/ארכימדס/י 571"/*.{pdf,PDF}; do
        if [[ -f "$img" ]]; then
            filename=$(basename "$img")
            if [[ "$filename" =~ מבוא ]]; then
                cp "$img" "$TARGET_DIR/5-10-archimedes-a/intro-1.pdf"
            elif [[ "$filename" =~ תוכן ]]; then
                if [[ "$filename" =~ ([0-9]+) ]]; then
                    num="${BASH_REMATCH[1]}"
                    cp "$img" "$TARGET_DIR/5-10-archimedes-a/toc-$num.pdf"
                fi
            fi
        fi
    done
    
    # Process cover image
    for img in "$SOURCE_DIR/ארכימדס/ארכימדס/י 571"/*.{png,PNG,jpg,JPG,jpeg}; do
        if [[ -f "$img" ]]; then
            extension="${img##*.}"
            cp "$img" "$TARGET_DIR/5-10-archimedes-a/cover-1.$extension"
        fi
    done
fi

# Grade 11, Level 5 (581)
if [[ -d "$SOURCE_DIR/ארכימדס/ארכימדס_ספר 581 - ינואר 2023" ]]; then
    echo "  Processing 5-11-archimedes volume..."
    mkdir -p "$TARGET_DIR/5-11-archimedes-a"
    
    # Process cover
    if [[ -f "$SOURCE_DIR/ארכימדס/ארכימדס_ספר 581 - ינואר 2023/Archimedes_Book_cover_581_new-1.png" ]]; then
        cp "$SOURCE_DIR/ארכימדס/ארכימדס_ספר 581 - ינואר 2023/Archimedes_Book_cover_581_new-1.png" "$TARGET_DIR/5-11-archimedes-a/cover-1.png"
    fi
    
    # Process M files as ToC pages
    counter=1
    for i in {1..10}; do
        if [[ -f "$SOURCE_DIR/ארכימדס/ארכימדס_ספר 581 - ינואר 2023/M$i.pdf" ]]; then
            cp "$SOURCE_DIR/ארכימדס/ארכימדס_ספר 581 - ינואר 2023/M$i.pdf" "$TARGET_DIR/5-11-archimedes-a/toc-$counter.pdf"
            ((counter++))
        fi
    done
fi

# Process יואל גבע (Yoel Geva) books
echo -e "\nProcessing יואל גבע books..."

# Grade 10, Level 3 (172)
if [[ -d "$SOURCE_DIR/יואל גבע/כיתה י 3 יחל/172" ]]; then
    echo "  Processing 3-10-geva volumes..."
    
    # Volume א
    if [[ -d "$SOURCE_DIR/יואל גבע/כיתה י 3 יחל/172/172_א" ]]; then
        mkdir -p "$TARGET_DIR/3-10-geva-a"
        for img in "$SOURCE_DIR/יואל גבע/כיתה י 3 יחל/172/172_א"/*.{JPG,jpg,jpeg,png,PNG}; do
            [[ -f "$img" ]] && process_image "$img" "$TARGET_DIR/3-10-geva-a"
        done
    fi
    
    # Volume ב
    if [[ -d "$SOURCE_DIR/יואל גבע/כיתה י 3 יחל/172/172_ב" ]]; then
        mkdir -p "$TARGET_DIR/3-10-geva-b"
        for img in "$SOURCE_DIR/יואל גבע/כיתה י 3 יחל/172/172_ב"/*.{JPG,jpg,jpeg,png,PNG}; do
            [[ -f "$img" ]] && process_image "$img" "$TARGET_DIR/3-10-geva-b"
        done
    fi
fi

# Grade 10, Level 4 (471)
if [[ -d "$SOURCE_DIR/יואל גבע/כיתה י 4 יחל" ]]; then
    echo "  Processing 4-10-geva volumes..."
    
    # Volume א
    if [[ -d "$SOURCE_DIR/יואל גבע/כיתה י 4 יחל/כרך_471_א" ]]; then
        mkdir -p "$TARGET_DIR/4-10-geva-a"
        for img in "$SOURCE_DIR/יואל גבע/כיתה י 4 יחל/כרך_471_א"/*.{JPG,jpg,jpeg,png,PNG}; do
            [[ -f "$img" ]] && process_image "$img" "$TARGET_DIR/4-10-geva-a"
        done
    fi
    
    # Volume ב
    if [[ -d "$SOURCE_DIR/יואל גבע/כיתה י 4 יחל/כרך_471_ב" ]]; then
        mkdir -p "$TARGET_DIR/4-10-geva-b"
        for img in "$SOURCE_DIR/יואל גבע/כיתה י 4 יחל/כרך_471_ב"/*.{JPG,jpg,jpeg,png,PNG}; do
            [[ -f "$img" ]] && process_image "$img" "$TARGET_DIR/4-10-geva-b"
        done
    fi
    
    # Volume ג
    if [[ -d "$SOURCE_DIR/יואל גבע/כיתה י 4 יחל/כרך_471_ג" ]]; then
        mkdir -p "$TARGET_DIR/4-10-geva-c"
        for img in "$SOURCE_DIR/יואל גבע/כיתה י 4 יחל/כרך_471_ג"/*.{JPG,jpg,jpeg,png,PNG}; do
            [[ -f "$img" ]] && process_image "$img" "$TARGET_DIR/4-10-geva-c"
        done
    fi
fi

# Grade 10, Level 5 (571, 804-806)
if [[ -d "$SOURCE_DIR/יואל גבע/כיתה י 5 יחל" ]]; then
    echo "  Processing 5-10-geva volumes..."
    
    # 571 volume
    if [[ -d "$SOURCE_DIR/יואל גבע/כיתה י 5 יחל/חוברת 571" ]]; then
        mkdir -p "$TARGET_DIR/5-10-geva-571"
        for img in "$SOURCE_DIR/יואל גבע/כיתה י 5 יחל/חוברת 571"/*.{JPG,jpg,jpeg,png,PNG}; do
            [[ -f "$img" ]] && process_image "$img" "$TARGET_DIR/5-10-geva-571"
        done
    fi
    
    # 804-806 Volume א
    if [[ -d "$SOURCE_DIR/יואל גבע/כיתה י 5 יחל/804-806_כרך_א" ]]; then
        mkdir -p "$TARGET_DIR/5-10-geva-804-a"
        for img in "$SOURCE_DIR/יואל גבע/כיתה י 5 יחל/804-806_כרך_א"/*.{JPG,jpg,jpeg,png,PNG}; do
            [[ -f "$img" ]] && process_image "$img" "$TARGET_DIR/5-10-geva-804-a"
        done
    fi
    
    # 804-806 Volume ב
    if [[ -d "$SOURCE_DIR/יואל גבע/כיתה י 5 יחל/804-806_כרך_ב" ]]; then
        mkdir -p "$TARGET_DIR/5-10-geva-804-b"
        for img in "$SOURCE_DIR/יואל גבע/כיתה י 5 יחל/804-806_כרך_ב"/*.{JPG,jpg,jpeg,png,PNG}; do
            [[ -f "$img" ]] && process_image "$img" "$TARGET_DIR/5-10-geva-804-b"
        done
    fi
fi

# Grade 11, Level 3 (371)
if [[ -d "$SOURCE_DIR/יואל גבע/כיתה יא 3 יחל" ]]; then
    echo "  Processing 3-11-geva volumes..."
    
    # Volume א
    if [[ -d "$SOURCE_DIR/יואל גבע/כיתה יא 3 יחל/כרך_371_א" ]]; then
        mkdir -p "$TARGET_DIR/3-11-geva-a"
        for img in "$SOURCE_DIR/יואל גבע/כיתה יא 3 יחל/כרך_371_א"/*.{JPG,jpg,jpeg,png,PNG}; do
            [[ -f "$img" ]] && process_image "$img" "$TARGET_DIR/3-11-geva-a"
        done
    fi
    
    # Volume ב
    if [[ -d "$SOURCE_DIR/יואל גבע/כיתה יא 3 יחל/כרך_371_ב" ]]; then
        mkdir -p "$TARGET_DIR/3-11-geva-b"
        for img in "$SOURCE_DIR/יואל גבע/כיתה יא 3 יחל/כרך_371_ב"/*.{JPG,jpg,jpeg,png,PNG}; do
            [[ -f "$img" ]] && process_image "$img" "$TARGET_DIR/3-11-geva-b"
        done
    fi
fi

# Grade 11, Level 4 (471)
if [[ -d "$SOURCE_DIR/יואל גבע/כיתה יא 4 יחל" ]]; then
    echo "  Processing 4-11-geva volumes..."
    
    # Volume א
    if [[ -d "$SOURCE_DIR/יואל גבע/כיתה יא 4 יחל/כרך_471_יא_א" ]]; then
        mkdir -p "$TARGET_DIR/4-11-geva-a"
        for img in "$SOURCE_DIR/יואל גבע/כיתה יא 4 יחל/כרך_471_יא_א"/*.{JPG,jpg,jpeg,png,PNG}; do
            [[ -f "$img" ]] && process_image "$img" "$TARGET_DIR/4-11-geva-a"
        done
    fi
    
    # Volume ב
    if [[ -d "$SOURCE_DIR/יואל גבע/כיתה יא 4 יחל/כרך_471_יא_ב" ]]; then
        mkdir -p "$TARGET_DIR/4-11-geva-b"
        for img in "$SOURCE_DIR/יואל גבע/כיתה יא 4 יחל/כרך_471_יא_ב"/*.{JPG,jpg,jpeg,png,PNG}; do
            [[ -f "$img" ]] && process_image "$img" "$TARGET_DIR/4-11-geva-b"
        done
    fi
fi

# Grade 11, Level 5 (806)
if [[ -d "$SOURCE_DIR/יואל גבע/כיתה יא 5 יחל" ]]; then
    echo "  Processing 5-11-geva volumes..."
    
    # Volume ג
    if [[ -d "$SOURCE_DIR/יואל גבע/כיתה יא 5 יחל/כרך_806_יא_ג" ]]; then
        mkdir -p "$TARGET_DIR/5-11-geva-c"
        for img in "$SOURCE_DIR/יואל גבע/כיתה יא 5 יחל/כרך_806_יא_ג"/*.{JPG,jpg,jpeg,png,PNG}; do
            [[ -f "$img" ]] && process_image "$img" "$TARGET_DIR/5-11-geva-c"
        done
    fi
    
    # Volume ד
    if [[ -d "$SOURCE_DIR/יואל גבע/כיתה יא 5 יחל/כרך_806_יא_ד" ]]; then
        mkdir -p "$TARGET_DIR/5-11-geva-d"
        for img in "$SOURCE_DIR/יואל גבע/כיתה יא 5 יחל/כרך_806_יא_ד"/*.{JPG,jpg,jpeg,png,PNG}; do
            [[ -f "$img" ]] && process_image "$img" "$TARGET_DIR/5-11-geva-d"
        done
    fi
fi

# Grade 12, Level 4 (805)
if [[ -d "$SOURCE_DIR/יואל גבע/כיתה יב 4 יחל/כרך 805 4 יחל" ]]; then
    echo "  Processing 4-12-geva volume..."
    mkdir -p "$TARGET_DIR/4-12-geva-a"
    for img in "$SOURCE_DIR/יואל גבע/כיתה יב 4 יחל/כרך 805 4 יחל"/*.{JPG,jpg,jpeg,png,PNG}; do
        [[ -f "$img" ]] && process_image "$img" "$TARGET_DIR/4-12-geva-a"
    done
fi

# Grade 12, Level 5 (807)
if [[ -d "$SOURCE_DIR/יואל גבע/כיתה יב 5 יחל" ]]; then
    echo "  Processing 5-12-geva volumes..."
    
    # Volume א
    if [[ -d "$SOURCE_DIR/יואל גבע/כיתה יב 5 יחל/כרך_807_יב_א" ]]; then
        mkdir -p "$TARGET_DIR/5-12-geva-a"
        for img in "$SOURCE_DIR/יואל גבע/כיתה יב 5 יחל/כרך_807_יב_א"/*.{JPG,jpg,jpeg,png,PNG}; do
            [[ -f "$img" ]] && process_image "$img" "$TARGET_DIR/5-12-geva-a"
        done
    fi
    
    # Volume ב
    if [[ -d "$SOURCE_DIR/יואל גבע/כיתה יב 5 יחל/כרך_807_יב_ב" ]]; then
        mkdir -p "$TARGET_DIR/5-12-geva-b"
        for img in "$SOURCE_DIR/יואל גבע/כיתה יב 5 יחל/כרך_807_יב_ב"/*.{JPG,jpg,jpeg,png,PNG}; do
            [[ -f "$img" ]] && process_image "$img" "$TARGET_DIR/5-12-geva-b"
        done
    fi
fi

# Generate summary report
echo -e "\n=== Organization Summary ==="
echo "Created directories in $TARGET_DIR:"
for dir in "$TARGET_DIR"/*; do
    if [[ -d "$dir" ]]; then
        count=$(find "$dir" -type f | wc -l)
        echo "  $(basename "$dir"): $count files"
    fi
done

echo -e "\nOrganization complete!"
echo "Note: Some files may have been skipped if they couldn't be categorized."
echo "Please review the $TARGET_DIR directory to ensure all volumes are correctly organized."