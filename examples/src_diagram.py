# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "asyncio",
#     "playwright<2",
# ]
# ///
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright
from typing import Optional

async def render_mermaid_to_png(mermaid_code: str, output_path: str, width: int = 1200, height: Optional[int] = None) -> None:
    """Render a Mermaid diagram to a PNG file using Playwright."""
    
    # Create HTML with the Mermaid code
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Mermaid Diagram</title>
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <script>
            mermaid.initialize({{
                startOnLoad: true,
                theme: 'default'
            }});
            
            // For detecting when rendering is complete
            window.addEventListener('load', function() {{
                window.renderedMermaid = false;
                setTimeout(() => {{
                    window.renderedMermaid = true;
                }}, 1000);
            }});
        </script>
        <style>
            body {{
                margin: 0;
                padding: 20px;
                display: flex;
                justify-content: center;
            }}
            .mermaid {{
                max-width: {width}px;
                margin: 0 auto;
            }}
        </style>
    </head>
    <body>
        <div class="mermaid">
{mermaid_code}
        </div>
    </body>
    </html>
    """
    
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        # Write HTML to a temporary file
        temp_html = Path("temp_mermaid.html")
        temp_html.write_text(html_content)
        
        # Navigate to the HTML file
        await page.goto(f"file://{temp_html.absolute()}")
        
        # Wait for Mermaid to render
        await page.wait_for_function("window.renderedMermaid === true")
        
        # Get the diagram element
        diagram = await page.query_selector(".mermaid svg")
        
        # Get the dimensions of the diagram
        box = await diagram.bounding_box()
        
        # Set the viewport size to match the diagram size plus some padding
        await page.set_viewport_size({
            "width": int(box["width"]) + 40,
            "height": int(box["height"]) + 40 if height is None else height
        })
        
        # Take screenshot of the diagram
        await diagram.screenshot(path=output_path)
        
        # Clean up
        await browser.close()
        temp_html.unlink()
        
        print(f"Diagram saved to {output_path}")

async def main():
    # Presentation diagram with layered architecture
    mermaid_code = """
graph TB
    subgraph "Input Layer"
        Input
    end
    
    subgraph "Problem Definition Layer"
        B[mesh_handling.jl]
        C[problem_definition.jl]
        D[problems.jl]
    end
    
    subgraph "Solver Layer"
        E[fem_solver.jl]
        G[nonlinear_solver.jl]
    end
    
    subgraph "Post-processing Layer"
        F[post_processing.jl]
        H[visualisation.jl]
    end
    
    subgraph "Output Layer"
        Output[VTK Files]
    end
    
    Input --> |"mesh"| B
    Input --> |"problem"| D
    B --> |"Mesh & Tags"| E
    C --> |"Material Properties"| E
    %% C --> |"Material Properties"| G
    D --> |"Weak Form"| E
    E --> |"Solution"| F
    E --> |"Solution"| G
    G --> |"Updated Properties"| D
    F --> |"Results"| H
    F --> |"Results"| Output

    style Output fill:#bfb,stroke:#333,stroke-width:2px
    style Input fill:#bfb,stroke:#333,stroke-width:2px
    """
    
    output_path = "magnetostatics_presentation.png"
    await render_mermaid_to_png(mermaid_code, output_path)

if __name__ == "__main__":
    asyncio.run(main())
