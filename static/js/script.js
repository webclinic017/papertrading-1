$(function(){

  const queryString = window.location.search;
  const urlParams = new URLSearchParams(queryString);
  const ts = urlParams.get('tradingsymbol')
  const autoload = urlParams.get('autoload')
  var img = new Image();
  img.onload = function() {
    image_height = parseInt(this.height);
    image_width = parseInt(this.width);
    image_reset_size();
    if (should_reset){
      image_reset_position();
    }
  };

  var end = moment().startOf('day').add(15, 'hour').add(30, 'minute');
  var start = end.clone().subtract(5, 'day').subtract(6, 'hour').subtract(16, 'minute');
  var current = start.clone();
  var tradingsymbol = (ts == null) ? 'NIFTY 50,NIFTY BANK' : ts
  var refresh = false
  var game_mode = false
  var buy = true
  var entry = 0
  var stoploss = 0
  var target = 0
  var lookback = 7
  var leadhr = 0
  var gameid = "unclassified"
  var intervalvar = null
  var tick_mode = false
  var image_height = 0
  var image_width = 0
  var should_reset = true
  var lasttime = null

  $('#current-date').val(current.toString());
  $('#lookback').val(lookback);
  $('#tradingsymbol').val(tradingsymbol);
  $('#sl').val(stoploss);
  $('#Entry').val(entry);
  $('#Target').val(target);
  $('#leadhr').val(leadhr);
  var wheight = $(window).height()-30
  $('.root-div').css("max-height", `${wheight}px`)
  $('.root-div').css("min-height", `${wheight}px`)

  $("#game-mode").change(function(){
      if($(this).is(":checked")) {
        $('#game-control').show()
        game_mode = true
      } else {
        $('#game-control').hide()
        game_mode = false

      }
  });

  $("#tick-mode").change(function(){
      if($(this).is(":checked")) {
        tick_mode = true
        $("#tradingsymbol").val("RELIANCE");
        tradingsymbol = "RELIANCE"
      } else {
        tick_mode = false
        $("#tradingsymbol").val("NIFTY 50,NIFTY BANK");
        tradingsymbol = "NIFTY 50,NIFTY BANK"
      }
  });

  $("#repeat").change(function(){
      if($(this).is(":checked")) {
        intervalvar = window.setInterval(schedule_refresh, 60000)
        console.log("starting refresh");
      } else {
        window.clearInterval(intervalvar)
        console.log('cleared refresh');
      }
  });

  function image_reset_size() {
    var maxheight = parseInt($(".root-div").css('height')) - 30;
    var newHeight = parseInt(maxheight > image_height ? image_height : maxheight);
    var newWidth = parseInt(maxheight > image_height ? image_width : image_width * (maxheight/image_height));

    $("#img-div").css("background-size", `${newWidth}px ${newHeight}px`);
    $('div.h-cross').css('width', newWidth + "px");
  }


  function image_reset_position() {
    // check if image with overflow to righ and then adjust
    var maxheight = parseInt($(".root-div").css('height')) - 30;
    var newHeight = parseInt(maxheight > image_height ? image_height : maxheight);
    var newWidth = parseInt(maxheight > image_height ? image_width : image_width * (maxheight/image_height));

    const xposition = Math.min(0, parseInt($(".root-div").css('width')) - newWidth);
    $("#img-div").css("background-position", `${xposition}px 10px, left top`);
    should_reset = false;
  }

  function stats_refresh(){
    $.ajax({
      headers: { "Accept": "application/json", "Access-Control-Allow-Headers": "*"},
      type: 'GET',
      url: 'stats/'+gameid,
      crossDomain: true,
      beforeSend: function(xhr){
          xhr.withCredentials = true;
    },
      success: function(data, textStatus, request){
          const obj = JSON.parse(data);
          $('#gameidview').val(obj["gameid"]);
          $('#accuracy').val(obj["accuracy"]);
          $('#change').val(obj["change"]);
          $('#count').val(obj["count"]);
          $('#maxdrawdown').val(obj["maxdrawdown"]);
          $('#partial').val(obj["partial"]);
          $('#peak').val(obj["peak"]);
      }
    });
  };

  function rrr_refresh(){
    var rrr = (parseFloat(target) - parseFloat(entry))/ (parseFloat(stoploss) - parseFloat(entry));
    rrr = Math.abs(rrr)
    rrr = rrr.toFixed(2)
    $('#rrr').val(rrr);
  };

  function day_change(){
    $.ajax({
      headers: { "Accept": "application/json", "Access-Control-Allow-Headers": "*"},
      type: 'GET',
      url: 'mis',
      crossDomain: true,
      beforeSend: function(xhr){
          xhr.withCredentials = true;
    },
      success: function(data, textStatus, request){
            data = JSON.parse(data);
            $("#dchange").val(data["dchange"]);
          },
      error: function(error){
        alert("Error, Failed to load graph")
      }
    });
  };

  function next_step(t1, t2, update){
    update = update ? 1: 0
    tm = tick_mode ? 1: 0
    lead = $("#lead").val()

    $.ajax({
      headers: { "Accept": "application/json", "Access-Control-Allow-Headers": "*"},
      type: 'GET',
      url: 'hgraph/'+tradingsymbol+'/'+t1+'/'+t2+"/"+update+"/"+tm+"/"+lead,
      crossDomain: true,
      beforeSend: function(xhr){
          xhr.withCredentials = true;
    },
      success: function(data, textStatus, request){
          data = JSON.parse(data);
          $("#img-div").css('background-image', `url('${data["imgurl"]}')`);
          img.src = data["imgurl"];
          if(data["update"] == "1"){
            $('#Entry').val(data["close"]);
            entry = data["close"];
            $('#Target').val(data["close"]);
            target = data["close"];
            $('#sl').val(data["close"]);
            stoploss = data["close"];
            rrr_refresh();
          }

      },
      error: function(error){
        alert("Error, Failed to load graph")
      }
    });
  };

  function schedule_refresh(){
    // console.log('refreshing');
    next_step(start.toString(), end.toString(), false);
    $('#last-updated').val(moment().format('MMMM Do YYYY, h:mm:ss a'));
    day_change();
  };


  $("#buy").change(function(){
    buy = $(this).is(":checked");
  })

  $("#tradingsymbol").change(function(){
    tradingsymbol = $(this).val();
    should_reset = true;
  })

  $("#Entry").change(function(){
    entry = $(this).val();
    rrr_refresh();
  })

  $("#Target").change(function(){
    target = $(this).val()
    rrr_refresh();
  })

  $("#sl").change(function(){
    stoploss = $(this).val()
    rrr_refresh();
  })

  $("#lookback").change(function(){
    lookback = $(this).val()
  })

  $("#gameid").change(function(){
    gameid = $(this).val()
  })

  $('#img-div').on('mousemove', function(e){
    $('div.h-cross').css('top', e.clientY-15);
  })

  $('#img-div').on('mouseenter', function(e){
    $('div.h-cross').css('visibility', 'visible');
  })

  $('#img-div').on('mouseleave', function(e){
    $('div.h-cross').css('visibility', 'hidden');
  })

  $('#clear-notes').on('click', function(){
    $('#notes').val('')
  });


  $("#graph-history").click(function(){
    next_step(start.toString(), end.toString(), false);
    $(document).prop('title', tradingsymbol + " | Lumbinicapital");
  });


  $('input[name="date-range"]').daterangepicker({
      timePicker: true,
      startDate: start,
      endDate: end,
      locale: {
        format: 'DD/MM/YYYY hh:mm A'
      }
    }, function(start_date, end_date){
      start = start_date
      end = end_date
      current = start_date.clone()
      $('#current-date').val(current.toString())
    });


  $('#Next-Hour').on('click', function(e){
    if(lasttime != null && (e.timeStamp - lasttime) < 10000){
      alert("Please wait atleast 10 second")
      return;
    }
    current = current.add(30, 'minute');
    $('#current-date').val(current.toString());

    var t1 = current.clone().subtract(lookback, "day").startOf('day').toString();
    var t2 = current.toString();

    next_step(t1, t2, true)

    leadhr = leadhr + 1;
    $(document).prop('title', tradingsymbol + " | Lumbinicapital");
    $('#leadhr').val(leadhr);
    lasttime = e.timeStamp;
  });


  $('#Review-Bet').on('click', function(){
    var fwd = 1
    var intraday = 1
    var bint = buy ? 1: 0

    var keyvar = {
      "name": tradingsymbol,
      "start": current.startOf('day').toString(),
      "fwd": fwd,
      "intraday": intraday,
      "leadhr": leadhr,
      "entry": entry,
      "stoploss": stoploss,
      "target": target,
      "buy": bint,
      "gameid": gameid
    }

    $.ajax({
      headers: { "Accept": "application/json", "Access-Control-Allow-Headers": "*"},
      type: 'POST',
      data: keyvar,
      url: 'outcome',
      crossDomain: true,
      beforeSend: function(xhr){
          xhr.withCredentials = true;
    },
      success: function(data, textStatus, request){
          $("#Change").val(data.change)
      }
    });
  });


  $('#reveal').on('click', function(){
    var t1 = current.clone().subtract(lookback, "day").startOf('day').toString();
    var t2 = current.clone().add(17, "hour").toString();
    next_step(t1, t2, false);
    stats_refresh();
  });


  $('#Next-Day').click(function(){
    var increment = 1
    if (current.format('E') == 5){
      increment = 3
    } else if (current.format('E') == 6) {
      increment = 2
    }
    current = current.add(increment, 'day').startOf('day').add(9, 'hour').add(14, 'minute');
    leadhr = 0;
    $('#current-date').val(current.toString());
    $('#leadhr').val(leadhr);
    $("#Change").val('undefined');
    $('#Entry').val(0);
    $('#sl').val(0);
    $('#Target').val(0);
    $(document).prop('title', tradingsymbol + " | Lumbinicapital");
    stats_refresh();
  });

  $('#Prev-Day').click(function(){
    var decrement = 1
    if (current.format('E') == 1){
      decrement = 3
    } else if (current.format('E') == 6) {
      decrement = 2
    }
    current = current.subtract(decrement, 'day').startOf('day').add(9, 'hour').add(14, 'minute');
    leadhr = 0;
    $('#current-date').val(current.toString());
    $('#leadhr').val(leadhr);
    $("#Change").val('undefined');
    $('#Entry').val(0);
    $('#sl').val(0);
    $('#Target').val(0);
    $(document).prop('title', tradingsymbol + " | Lumbinicapital");
    stats_refresh();
  });

  $('#reset-graph').on("click", function(){
    image_reset_size();
    image_reset_position();
   });

    if(autoload != null){
      schedule_refresh();
      $(document).prop('title', tradingsymbol + " | Lumbinicapital");
    };


});
