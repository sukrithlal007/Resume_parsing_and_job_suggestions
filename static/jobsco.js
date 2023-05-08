
        $(document).on('submit','#suggestForm', function(e){
            e.preventDefault();
            console.log("Skrsdf")
            const skillSet = $("#skillSet");
            const suggestedJobs = $("#suggestedJobs");

            const suggestedJobsTable = $("#results");
            // const loader = $("#loader");
            const sp  = $("#showpiece")
            const submitBtn = $("#submitBtn");

            skillSet.hide();
            suggestedJobsTable.hide();
            submitBtn.hide();
            sp.show()
            // loader.show();

            let formData = new FormData($(this)[0]);
            let formUrl = $(this).attr('action');
            $.ajax({
                url: formUrl,
                data: formData,
                type: 'POST',
                processData: false,
                contentType: false,

            }).done(function(data){
                if(data.status == 500){
                    alert(data.msg);
                     submitBtn.show();
                    loader.hide();
                    sp.hide()
                    suggestedJobsTable.hide();
                    return;
                }

                let skills = data.skills;
                let jobs = data.matchedJobsList;

                let skillsHtml = "";
                let jobsHtml = "";
                skillsHtml += skills.map(skill => " "+skill);
                skillSet.html(skillsHtml);

                for(var i = 0; i < jobs.length; i++){
                console.log(jobs[i]);
                    jobsHtml += `<tr>`;
                    jobsHtml += `<td>${i+1}</td>`;
                    jobsHtml += `<td>${jobs[i][1]}</td>`;
                    jobsHtml += `<td>${jobs[i][2]}</td>`;
                    jobsHtml += `<td>${jobs[i][3]}</td>`;
                    jobsHtml += `<td>${jobs[i][4]}</td>`;
                    jobsHtml += `</tr>`;
                }

                suggestedJobs.html(jobsHtml);

                submitBtn.show();
                // loader.hide();
                sp.hide()
                suggestedJobsTable.show();
                skillSet.show();
            }).fail(function(){
                alert("Something went wrong");
                submitBtn.show();
                // loader.hide();
                sp.hide();
                suggestedJobsTable.show();
                skillSet.show();

            })

        })
