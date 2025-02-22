document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("search-btn").addEventListener("click", function () {
        const query = document.getElementById("query").value;

        if (query) {
            fetch(`/search?query=${encodeURIComponent(query)}`)
                .then(response => response.json())
                .then(data => {
                    const resultsDiv = document.getElementById("results");
                    resultsDiv.innerHTML = "<h2>🔍 Top 5 Matching Profiles</h2>";

                    if (data.length === 0) {
                        resultsDiv.innerHTML += "<p>No matching profiles found.</p>";
                    } else {
                        data.forEach((result, index) => {
                            const card = document.createElement("div");
                            card.className = "result-card";
                            card.innerHTML = `
                                <h3>Result ${index + 1}</h3>
                                <p><strong>👤 Name:</strong> ${result.Name}</p>
                                <p><strong>💼 Title:</strong> ${result.Title}</p>
                                <p><strong>🛠 Skills:</strong> ${result.Skills}</p>
                                <p><strong>📆 Experience:</strong> ${result.Experience}</p>
                                <p><strong>📍 Location:</strong> ${result.Location}</p>
                            `;
                            resultsDiv.appendChild(card);
                        });
                    }
                })
                .catch(error => {
                    console.error("Error:", error);
                });
        } else {
            alert("Please enter a search query.");
        }
    });
});
