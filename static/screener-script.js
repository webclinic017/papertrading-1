$(document).ready(function() {
    var rscore = $('#rscore').DataTable();
    var date = moment().startOf('day').format('YYYY-MM-DD');
    $('#date').val(date.toString());
    $("#date").change(function(){
      date = $(this).val()
      console.log(date);
    })

    function stats_data(){
            $.ajax({
              headers: { "Accept": "application/json", "Access-Control-Allow-Headers": "*"},
              type: 'GET',
              url: 'ss'+"/"+date,
              crossDomain: true,
              beforeSend: function(xhr){
                  xhr.withCredentials = true;
            },
              success: function(data, textStatus, request){
                  const obj = JSON.parse(data);
                  for (let i = 0; i < obj["rscore"].length; i++) {
                    rscore.row.add(obj["rscore"][i]).draw( false );
                  }
                  $("#status").val("Idle")
      }})};

      // intervalvar = window.setInterval(stats_data, 60000)

      $("#refresh").on("click", function(){
        $("#status").val("Fetching...")
        rscore.rows().remove().draw();
        stats_data();
      })
});
