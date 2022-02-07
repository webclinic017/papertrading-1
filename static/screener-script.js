$(document).ready(function() {
    var rscore = $('#rscore').DataTable();

    function stats_data(){
            $.ajax({
              headers: { "Accept": "application/json", "Access-Control-Allow-Headers": "*"},
              type: 'GET',
              url: 'ss',
              crossDomain: true,
              beforeSend: function(xhr){
                  xhr.withCredentials = true;
            },
              success: function(data, textStatus, request){
                  const obj = JSON.parse(data);
                  rscore.rows().remove().draw();
                  for (let i = 0; i < obj["rscore"].length; i++) {
                    rscore.row.add(obj["rscore"][i]).draw( false );
                  }
                  $("#status").val("Idle")
      }})};

      // intervalvar = window.setInterval(stats_data, 60000)

      $("#refresh").on("click", function(){
        $("#status").val("Fetching...")
        stats_data();
      })
});
